"""边缘优化与后处理（对应技术方案 5.4）。

包含：
- feather     羽化，消除硬锯齿
- erode_alpha 收缩，去掉背景残留光晕
- denoise     去孤立噪点（按连通域面积过滤）
- despill     去白边/色污染（合成公式反解前景色）
- merge_alpha 合并输入图片自带的 alpha 与计算得到的 alpha
- extend_edge_color 边缘颜色外扩，避免透明区黑边（用于"沿用已有 alpha"场景）
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def feather(alpha: np.ndarray, radius: int = 1) -> np.ndarray:
    """对 alpha 做高斯模糊，使边缘平滑过渡。"""
    if radius <= 0:
        return alpha
    ksize = radius * 2 + 1
    return cv2.GaussianBlur(alpha, (ksize, ksize), 0)


def erode_alpha(alpha: np.ndarray, px: int = 1) -> np.ndarray:
    """腐蚀 alpha，向内收缩，用于消除背景色光晕。"""
    if px <= 0:
        return alpha
    ksize = px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(alpha, kernel)


def denoise(alpha: np.ndarray, min_area: int = 32, min_area_ratio: float = 0.0) -> np.ndarray:
    """去除面积过小的孤立前景块（视为噪点），置为透明。

    min_area_ratio 按图像面积比例设定下限（取两者较大值），
    避免小图被误清空。
    """
    if min_area <= 0 and min_area_ratio <= 0:
        return alpha

    total = alpha.size
    area_thr = max(min_area, int(total * min_area_ratio))

    fg = (alpha >= 128).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1:
        return alpha

    out = alpha.copy()
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] < area_thr:
            out[labels == label] = 0
    return out


def despill(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg: np.ndarray,
    min_alpha: float = 0.2,
) -> np.ndarray:
    """去色污染（去白边/黑边）。

    合成公式 C = a·F + (1-a)·B，反解前景色 F = (C - (1-a)·B) / a。
    只对半透明边缘像素做处理，并给 alpha 设下限避免数值爆炸。
    背景为白色时即为经典的"去白边"。
    """
    a = alpha.astype(np.float32) / 255.0
    bg = np.asarray(bg, dtype=np.float32)

    edge = (a > 0.01) & (a < 0.99)
    if not np.any(edge):
        return rgb

    out = rgb.astype(np.float32).copy()
    ae = a[edge][:, None]
    ae_safe = np.clip(ae, min_alpha, 1.0)
    fg = (rgb[edge].astype(np.float32) - (1.0 - ae) * bg[None, :]) / ae_safe
    out[edge] = np.clip(fg, 0.0, 255.0)
    return out.astype(np.uint8)


def harden_alpha(
    alpha: np.ndarray,
    lo: int = 16,
    hi: int = 240,
) -> np.ndarray:
    """alpha 硬化：接近两端的值吸附到 0 / 255，中间过渡带保留。

    部分已抠好的 PNG 主体 alpha 是 254 而非 255（实测 deepseek4.png 占 58.5%），
    在少数软件里会被当成半透明处理。硬化既修正了这一点，又不破坏边缘
    真实存在的半透明过渡像素（位于 lo~hi 之间的原样保留）。
    """
    out = alpha.astype(np.uint8).copy()
    out[out <= lo] = 0
    out[out >= hi] = 255
    return out


def extend_edge_color(
    rgb: np.ndarray,
    alpha: np.ndarray,
    iterations: int = 8,
) -> np.ndarray:
    """把边缘颜色向外扩散，填充透明区域的 RGB（防黑边）。

    已经抠好的 PNG，透明区下的 RGB 常被压成纯黑。若不处理，后续做缩放、
    羽化或二次合成时，这些黑色会渗进边缘形成黑边。
    本函数逐轮把"已知颜色"向未知区域平均扩散一轮（每轮 1 像素），
    只改写 alpha==0 的区域，不影响任何可见像素。
    """
    unknown = alpha <= 0
    if not unknown.any():
        return rgb
    if unknown.all():
        return rgb

    out = rgb.astype(np.float32)
    known = ~unknown
    for _ in range(max(1, iterations)):
        if not unknown.any():
            break
        kf = known.astype(np.float32)
        weight = cv2.blur(kf, (3, 3))
        acc = np.stack(
            [cv2.blur(out[:, :, c] * kf, (3, 3)) for c in range(3)],
            axis=2,
        )
        newval = acc / np.maximum(weight[:, :, None], 1e-6)
        updatable = unknown & (weight > 0)
        if not updatable.any():
            break
        out[updatable] = newval[updatable]
        known |= updatable
        unknown &= ~updatable

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def merge_alpha(computed: np.ndarray, origin: Optional[np.ndarray]) -> np.ndarray:
    """若原图自带 alpha，则与计算得到的 alpha 相乘合并。"""
    if origin is None:
        return computed
    merged = computed.astype(np.uint16) * origin.astype(np.uint16) // 255
    return merged.astype(np.uint8)


def refine(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg: np.ndarray,
    feather_radius: int = 1,
    erode_px: int = 0,
    despill_on: bool = True,
    denoise_min_area: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """串起后处理步骤，返回 (处理后的 rgb, 处理后的 alpha)。"""
    a = alpha
    if denoise_min_area > 0:
        a = denoise(a, min_area=denoise_min_area)
    if erode_px > 0:
        a = erode_alpha(a, erode_px)
    if despill_on:
        rgb = despill(rgb, a, bg)
    if feather_radius > 0:
        a = feather(a, feather_radius)
    return rgb, a
