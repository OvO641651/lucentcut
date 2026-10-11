"""纯色背景抠图（白底 / 黑底 / 任意纯色底）。

算法流程（对应技术方案 5.1）：
  1. 边框采样 + 中位数估计背景色，并据此判定背景类型；
  2. 计算每个像素与背景色的欧氏距离；
  3. 软阈值把距离映射为 0~255 的 alpha（过渡带实现平滑边缘）；
  4. 连通域修正：仅把与图像边界连通的透明区视为真背景，
     主体内部的白色块（如白色文字/反光）恢复为不透明；
  5. 交给 refine 模块做羽化/收缩/去白边。

本模块只依赖 numpy + opencv，不涉及任何 AI 模型。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

WHITE = "white"
BLACK = "black"
SOLID = "solid"      # 任意纯色背景（灰/蓝/绿等非黑白纯色）
COMPLEX = "complex"

# 背景色判定阈值
_WHITE_MIN = 225     # 边框中位亮度 >= 该值 → 白底
_BLACK_MAX = 45      # 边框中位亮度 <= 该值 → 黑底
_INLIER_DIST = 45.0  # 边框像素与背景中位色的"内点"距离阈值
_INLIER_MIN = 0.55   # 内点占比低于该值 → 判定为复杂背景


def _border_pixels(rgb: np.ndarray, thickness: int = 3) -> np.ndarray:
    """取图片四条边框的像素（形状 N×3）。"""
    h, w = rgb.shape[:2]
    t = max(1, min(thickness, h // 2, w // 2))
    top = rgb[:t, :, :].reshape(-1, 3)
    bottom = rgb[-t:, :, :].reshape(-1, 3)
    left = rgb[:, :t, :].reshape(-1, 3)
    right = rgb[:, -t:, :].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def estimate_background(rgb: np.ndarray, thickness: int = 3) -> np.ndarray:
    """用边框像素的逐通道中位数估计背景色，返回 (3,) float32。

    用中位数而非均值，可避免主体占据边角时被拉偏；也比只取四角更稳健。
    """
    border = _border_pixels(rgb, thickness).astype(np.float32)
    return np.median(border, axis=0)


def detect_mode(rgb: np.ndarray, thickness: int = 3) -> str:
    """自动识别背景类型：white / black / solid / complex。

    思路：先用边框中位色作为背景代表色，再统计边框像素中"接近该代表色"
    的内点占比。只要大部分边框是同一个纯色（即使主体占了某个边角），
    就视为纯色背景；纯色又按亮度细分为白底 / 黑底 / 其它纯色（solid）；
    否则视为复杂背景。
    """
    border = _border_pixels(rgb, thickness).astype(np.float32)
    median = np.median(border, axis=0)

    inlier_dist = np.sqrt(np.sum((border - median) ** 2, axis=1))
    inlier_ratio = float(np.mean(inlier_dist < _INLIER_DIST))
    brightness = float(median.mean())

    if inlier_ratio < _INLIER_MIN:
        return COMPLEX
    if brightness >= _WHITE_MIN:
        return WHITE
    if brightness <= _BLACK_MAX:
        return BLACK
    return SOLID


def color_distance(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """每个像素与背景色的欧氏距离（float32, 形状 H×W）。"""
    diff = rgb.astype(np.float32) - bg.astype(np.float32)[None, None, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def soft_threshold(dist: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """软阈值：距离 <= t_low → 0；>= t_high → 255；之间线性过渡。"""
    if t_high <= t_low:
        raise ValueError("t_high 必须大于 t_low")
    alpha = (dist - t_low) / (t_high - t_low)
    alpha = np.clip(alpha, 0.0, 1.0) * 255.0
    return alpha.astype(np.uint8)


def keep_border_connected(alpha: np.ndarray, thr: int = 250) -> np.ndarray:
    """仅保留与图像边界连通的透明区域为真背景。

    主体内部的白色/背景色区域（不与边界连通）会被恢复为不透明，
    避免在主体上"抠出洞"。

    thr 必须取得**足够接近 255**（默认 250），而不是 128。原因：
    soft_threshold 会在轮廓上生成一整条 [1,254] 的过渡带，若只把 <128 视为
    背景候选，主体内部那些"浅色/低对比"的部件就会掉进 128~249 之间——
    既不算背景、也不算前景，被永久留成半透明，合成出来是一块发灰发噪的脏斑。

    （实测 deepseek.jpg 的眼睛高光：RGB≈(205,223,239)，到白底色差仅 55~79，
      落在软阈值 [30,80] 内 → alpha≈160，被深蓝眼珠完全包住却不被修复。
      放大后即肉眼可见的"画质下降"。）

    判据：**背景 = 与图像边界连通的那一块**；不属于它的非全不透明像素一律
    判为主体内部，置为 255。
    """
    bg_candidate = (alpha < thr).astype(np.uint8)
    if bg_candidate.max() == 0:
        return alpha

    num, labels = cv2.connectedComponents(bg_candidate, connectivity=8)
    if num <= 1:
        return alpha

    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ]))
    border_labels = border_labels[border_labels != 0]

    in_bg = np.isin(labels, border_labels)
    # 不属于"边界连通背景"的非全不透明像素 → 判定为主体内部，恢复不透明
    restore = (~in_bg) & (alpha < 255)

    out = alpha.copy()
    out[restore] = 255
    return out


def remove_solid_bg(
    rgb: np.ndarray,
    mode: str = "auto",
    t_low: float = 30.0,
    t_high: float = 80.0,
    border_thickness: int = 3,
    bg_color: Optional[np.ndarray] = None,
    fix_holes: bool = True,
) -> Tuple[np.ndarray, str, np.ndarray]:
    """去除纯色背景，返回 (alpha, 实际使用的模式, 背景色)。

    参数
    ----
    mode : white / black / solid / complex / auto
    t_low, t_high : 软阈值上下界（0~441 的色差范围）
    bg_color : 手动指定背景色（覆盖自动估计）
    fix_holes : 是否做边界连通域修正
    """
    if mode == "auto":
        mode = detect_mode(rgb, border_thickness)

    if mode == COMPLEX:
        raise ValueError("检测到复杂背景，请使用 AI 分割或交互式抠图模式")

    bg = estimate_background(rgb, border_thickness) if bg_color is None \
        else np.asarray(bg_color, dtype=np.float32)

    dist = color_distance(rgb, bg)
    alpha = soft_threshold(dist, t_low, t_high)

    if fix_holes and alpha.size > 0:
        alpha = keep_border_connected(alpha)

    return alpha, mode, bg
