"""交互式抠图（M7）+ 人工裁决层（M9）。

两种语义（M10 起明确区分，见技术方案 5.4 / 5.7）：

A. **直接裁决**（keep / drop，GUI 的默认路径，不做任何算法计算）
- 点选 points：按住拖动涂抹圆盘，涂在哪就精确影响哪一块（points_to_mask）
- rect / polygon / lasso：圈出闭合区域（regions_to_mask）
  它们统一写进 keep_mask / drop_mask，由 apply_override 叠加到算法结果上。

B. **GrabCut 自动抠图**（GUI 里是一个独立按钮，不再与"保留/去除"混在一个下拉框）
- rect     矩形框选：拉一个框圈住主体，GrabCut 自动分离
- polygon  描边：沿轮廓点一圈，区域内作为前景先验
- lasso    套索：自由曲线闭合区域
- points   点选种子：fg=前景种子、bg=背景种子（保留为底层 API，GUI 不再调用）

底层用 OpenCV GrabCut（无需 AI 模型），对主体与背景颜色差异明显的场景效果最好。
所有方式最终都返回 (rgb_out, alpha)，alpha 已做羽化去锯齿。

M9 起区域统一用「多边形点列」表示，可一次提交**多个互不相连的区域**：
- regions_to_mask(shape, regions)         多区域栅格化
- matting_from_regions(rgb, regions)      多区域 + GrabCut
- apply_override(...)                     人工裁决：框内保留 / 框内去除

接口：
- points_to_mask(shape, points, radius) 点选笔刷 → 圆盘掩码（直接裁决）
- matting_from_rect(rgb, rect)          rect = (x, y, w, h)
- matting_from_points(rgb, fg_pts, bg_pts)
- matting_from_polygon(rgb, polygon)    polygon = [(x, y), ...]（闭合）
- matting_from_mask(rgb, mask)          mask: uint8，255=确定前景
- rect_to_region(rect)                  矩形 → 4 点区域，用于统一到多区域流程
- regions_to_mask(shape, regions)       区域列表 → 二值掩码（255=区域内）
- matting_from_regions(rgb, regions)    多区域 GrabCut
- apply_override(rgb, alpha, origin_rgb, keep_mask, drop_mask, feather_px)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import refine


def _grabcut(rgb: np.ndarray, mask_hint: Optional[np.ndarray] = None,
             rect: Optional[Tuple[int, int, int, int]] = None,
             iterations: int = 5) -> np.ndarray:
    """执行 GrabCut，返回二值前景 mask（uint8 0/255）。

    mask_hint: 可选，uint8 掩码，标记已知前景(255)/背景(0)，其余未知。
    rect: 可选，包围盒 (x, y, w, h)，作为前景先验。
    """
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    if mask_hint is not None:
        # GC_FGD=1, GC_BGD=0, GC_PR_FGD=3（确定前景）, GC_PR_BGD=2
        gc_mask = np.full(rgb.shape[:2], cv2.GC_PR_BGD, np.uint8)  # 默认可能背景
        gc_mask[mask_hint == 255] = cv2.GC_PR_FGD
        gc_mask[mask_hint == 1] = cv2.GC_BGD
        rect = None
        mode = cv2.GC_INIT_WITH_MASK
    else:
        gc_mask = np.zeros(rgb.shape[:2], np.uint8)
        mode = cv2.GC_INIT_WITH_RECT

    cv2.grabCut(rgb, gc_mask, rect, bgd, fgd, iterations, mode)
    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg


def _finalize(rgb: np.ndarray, fg: np.ndarray, feather_px: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """把二值前景 mask 转成带羽化的 alpha，并返回处理后的 RGB。"""
    alpha = refine.feather(fg, feather_px)
    bg = np.array([0, 0, 0], dtype=np.float32)  # 去污染用背景取黑（交互式无固定背景色）
    rgb_out = rgb
    return rgb_out, alpha


def matting_from_rect(rgb: np.ndarray, rect: Tuple[int, int, int, int],
                      feather_px: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """矩形框选抠图。rect=(x,y,w,h)。"""
    x, y, w, h = rect
    if w < 2 or h < 2:
        raise ValueError("框选区域过小")
    fg = _grabcut(rgb, rect=rect)
    return _finalize(rgb, fg, feather_px)


def matting_from_points(rgb: np.ndarray, fg_pts: Sequence[Tuple[int, int]],
                        bg_pts: Sequence[Tuple[int, int]],
                        feather_px: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """点选抠图。fg_pts=前景种子点，bg_pts=背景种子点（可都为空则退化为矩形全图）。"""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for x, y in fg_pts:
        cv2.circle(mask, (int(x), int(y)), 3, 255, -1)
    for x, y in bg_pts:
        cv2.circle(mask, (int(x), int(y)), 3, 1, -1)  # 1 表示确定背景标记
    if not fg_pts and not bg_pts:
        fg = _grabcut(rgb, rect=(0, 0, rgb.shape[1], rgb.shape[0]))
    else:
        fg = _grabcut(rgb, mask_hint=mask)
    return _finalize(rgb, fg, feather_px)


def matting_from_polygon(rgb: np.ndarray, polygon: Sequence[Tuple[int, int]],
                         feather_px: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """描边（多边形）抠图。polygon 为顶点列表，自动闭合。"""
    if len(polygon) < 3:
        raise ValueError("多边形至少需要 3 个顶点")
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    fg = _grabcut(rgb, mask_hint=mask)
    return _finalize(rgb, fg, feather_px)


def matting_from_mask(rgb: np.ndarray, mask: np.ndarray,
                      feather_px: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """直接用二值 mask 抠图（套索等自由曲线绘制后先栅格化成 mask 再调用）。"""
    fg = _grabcut(rgb, mask_hint=mask)
    return _finalize(rgb, fg, feather_px)


def rasterize_curve(rgb_shape: Tuple[int, int], points: Sequence[Tuple[int, int]],
                    thickness: int = 1) -> np.ndarray:
    """把自由曲线点列栅格化成闭合二值 mask（用于套索）。

    曲线按点序连线并闭合，内部填 255。
    """
    h, w = rgb_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if len(points) < 3:
        return mask
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# --------------------------------------------------------------------------- #
# M9：多区域 + 人工裁决层
# --------------------------------------------------------------------------- #
def rect_to_region(rect: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """矩形 (x, y, w, h) → 4 点多边形，统一进多区域流程。"""
    x, y, w, h = rect
    return [(int(x), int(y)), (int(x + w), int(y)),
            (int(x + w), int(y + h)), (int(x), int(y + h))]


def regions_to_mask(shape: Tuple[int, ...],
                    regions: Sequence[Sequence[Tuple[int, int]]],
                    value: int = 255) -> np.ndarray:
    """把多个多边形区域栅格化成一张二值掩码。

    每个区域独立填充，互不相连也能正确表达（解决"一次只能描一个区域"的问题）。
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for pts in regions:
        if len(pts) < 3:
            continue
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [arr], int(value))
    return mask


def matting_from_regions(rgb: np.ndarray,
                         regions: Sequence[Sequence[Tuple[int, int]]],
                         feather_px: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """多个区域作为前景先验，一次 GrabCut 求解（可处理多个互不相连的主体）。"""
    mask = regions_to_mask(rgb.shape, regions)
    if mask.max() == 0:
        raise ValueError("没有任何有效区域，请先描出至少一个闭合区域")
    return matting_from_mask(rgb, mask, feather_px)


def points_to_mask(shape: Tuple[int, ...],
                   points: Sequence[Tuple[int, int]],
                   radius: int = 12,
                   value: int = 255) -> np.ndarray:
    """点选（笔刷）坐标 → 圆盘掩码。半径按**原图像素**计。

    与 matting_from_points（GrabCut 种子）不同，这里不做任何分割计算：
    点在哪里、半径多大，就精确地影响哪一块像素。这是 M10 之后「点选」
    的语义——用户要的是"我这块白色眼睛必须留下"，而不是"让算法猜"。
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    r = max(1, int(radius))
    for x, y in points:
        cv2.circle(mask, (int(x), int(y)), r, int(value), -1)
    return mask


def merge_override_masks(
    keep: Optional[np.ndarray],
    drop: Optional[np.ndarray],
    keep_new: Optional[np.ndarray],
    drop_new: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """跨笔合并人工裁决层（M13，决策 D13 修订）：**后画的覆盖先画的**。

    keep / drop          既有裁决掩码（None = 还没有），入参不会被修改
    keep_new / drop_new  本次新画的掩码（uint8，0/255）

    规则
    ----
    1. 新去除会撤销同一位置的旧保留；新保留会撤销同一位置的旧去除。
       —— 修复"先保留过的区域，后画的去除永远不生效"的缺陷（旧实现
       无条件 `drop[keep>0]=0`，等于保留一旦画下就永久赢）。
    2. 同一笔提交里既画保留又画去除且重叠时，以保留为准（防误删）；
       这与 apply_override 的叠加顺序（先 drop 后 keep）一致。
    """
    keep = None if keep is None else keep.copy()
    drop = None if drop is None else drop.copy()
    if drop_new is not None and drop_new.any():
        if keep is not None:
            keep[drop_new > 0] = 0
        drop = drop_new.copy() if drop is None else np.maximum(drop, drop_new)
    if keep_new is not None and keep_new.any():
        if drop is not None:
            drop[keep_new > 0] = 0
        keep = keep_new.copy() if keep is None else np.maximum(keep, keep_new)
    return keep, drop


def apply_override(
    rgb: np.ndarray,
    alpha: np.ndarray,
    origin_rgb: Optional[np.ndarray] = None,
    keep_mask: Optional[np.ndarray] = None,
    drop_mask: Optional[np.ndarray] = None,
    feather_px: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """人工裁决层：把用户描出的"必须保留 / 必须去除"区域叠加到算法结果上。

    设计要点（决策 D10）
    ------------------
    1. 裁决层**优先于算法**：先施加 drop（置 0）再施加 keep（置 255），
       两者重叠时以"保留"为准——用户专门描出来要留的东西，不该被另一笔误删。
    2. keep 区域若原本被算法判成背景（alpha < 128），用**原图 RGB** 把颜色补回来。
       算法在透明区可能已做过边缘色外扩/despill，直接沿用会留下脏色，
       而用户要的是"原图中衣服本来的颜色"。
    3. 只在裁决层自己的边界做羽化，不改动画面其它地方，避免整图被糊一遍。

    返回 (rgb_out, alpha_out)。
    """
    if keep_mask is None and drop_mask is None:
        return rgb, alpha

    out = alpha.astype(np.uint8).copy()
    out_rgb = rgb
    if origin_rgb is None:
        origin_rgb = rgb

    union = np.zeros(alpha.shape[:2], np.uint8)

    if drop_mask is not None and drop_mask.any():
        d = drop_mask > 0
        out[d] = 0
        union |= d.astype(np.uint8)

    if keep_mask is not None and keep_mask.any():
        k = keep_mask > 0
        # 原图颜色回填：只回填"算法原本判成背景"的像素，
        # 本来就是前景的地方保留算法处理后的（可能已去白边）颜色。
        restored = k & (alpha < 128)
        if restored.any():
            if out_rgb is rgb:
                out_rgb = rgb.copy()
            out_rgb[restored] = origin_rgb[restored]
        out[k] = 255
        union |= k.astype(np.uint8)

    if feather_px > 0 and union.any():
        out = _feather_override_edge(out, union, feather_px)

    return out_rgb, out


def _feather_override_edge(alpha: np.ndarray, union: np.ndarray,
                           feather_px: int) -> np.ndarray:
    """只在裁决层边界的窄带内做羽化，区域内部保持满值（255/0）。

    若直接对整图做高斯模糊，保留区最内一圈会被糊成 ~50% 半透明，
    看上去像"保留是假的"。这里因此分三层处理：
    内部（腐蚀掉边界）保持硬值 → 边界窄带做渐变 → 带外原样不动。
    """
    k3 = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(union, cv2.MORPH_GRADIENT, k3)
    if grad.max() == 0:  # 全图被裁决层覆盖时没有边界可柔化
        return alpha
    ksize = feather_px * 2 + 1
    band = cv2.dilate(grad, np.ones((ksize, ksize), np.uint8))
    blurred = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    out = np.where(band > 0, blurred, alpha).astype(np.uint8)
    # 内部（离边界 >= feather+1 像素）恢复硬值
    inner = cv2.erode(union, np.ones((ksize + 2, ksize + 2), np.uint8))
    if inner.any():
        out[inner > 0] = alpha[inner > 0]
    return out
