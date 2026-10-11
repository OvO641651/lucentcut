"""图片读写模块。

统一以 numpy 数组在内部传递：
- rgb   : uint8, 形状 (H, W, 3)
- alpha : uint8, 形状 (H, W)，255=不透明，0=全透明
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # 允许处理超大图，交由调用方控制


def load_image(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """读取图片，返回 (rgb, alpha)。

    alpha 为 None 表示原图没有透明通道。

    说明（2026-10-11 修复）：手机拍摄的 JPEG 常带 EXIF Orientation 标记，
    PIL 默认不会据此旋转像素，直接读出来会是横躺/倒置的。这里统一用
    `ImageOps.exif_transpose` 把画面摆正后再取像素，避免输出方向错误。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")

    try:
        im = Image.open(path)
        im.load()
        im = ImageOps.exif_transpose(im) or im
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法识别或已损坏的图片: {path} ({exc})") from exc

    has_alpha = "A" in im.getbands()
    rgba = im.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    rgb = np.ascontiguousarray(arr[:, :, :3])
    alpha = np.ascontiguousarray(arr[:, :, 3]) if has_alpha else None
    return rgb, alpha


def save_rgba(rgb: np.ndarray, alpha: np.ndarray, path: str) -> str:
    """把 rgb + alpha 合成为 RGBA PNG 并保存，返回输出路径。"""
    if rgb.shape[:2] != alpha.shape[:2]:
        raise ValueError("rgb 与 alpha 尺寸不一致")

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")
    return path


def alpha_is_trimmed(
    alpha: np.ndarray,
    binary_ratio_thr: float = 0.98,
    lo: int = 32,
    hi: int = 224,
    border_transparent_thr: float = 0.90,
    border_thickness: int = 3,
) -> Tuple[bool, dict]:
    """判断输入图片的 alpha 是否已经是"抠好的透明图"。

    背景（对应技术方案 5.1 Step 0）：
    已经抠好的 PNG，其透明区下方的 RGB 通常被压成纯黑（保存机制所致），
    若仍按颜色去抠，会把主体上偏暗的部分当成背景删掉——这是 M1 实测发现
    的一个真实缺陷（详见文档 12.2）。因此凡检测到"干净且完整的 alpha"，
    就应当直接沿用，不再做颜色键控。

    判定条件（两条同时满足才算已抠好）：
      1. 二值性：alpha 落在 (<lo, >hi) 的像素占比 >= binary_ratio_thr
         —— 抠好的图 alpha 近乎非 0 即 255，几乎没有半透明胶水层；
      2. 边框透明：四条边框(alpha<16)占比 >= border_transparent_thr
         —— 说明背景确实已被掏空，而不是只抠了局部。

    返回 (是否可直接沿用, 诊断信息 dict)。
    """
    a = np.asarray(alpha)
    h, w = a.shape
    total = a.size
    if total == 0:
        return False, {"reason": "空图像"}

    binary_ratio = float(np.mean((a < lo) | (a > hi)))

    t = max(1, min(border_thickness, h // 2, w // 2))
    border = np.concatenate([
        a[:t, :].ravel(), a[-t:, :].ravel(),
        a[:, :t].ravel(), a[:, -t:].ravel(),
    ])
    border_transparent = float(np.mean(border < 16))
    transparent_ratio = float(np.mean(a < 16))

    info = {
        "binary_ratio": binary_ratio,
        "border_transparent": border_transparent,
        "transparent_ratio": transparent_ratio,
        "semi_transparent_ratio": float(np.mean((a >= 16) & (a <= 239))),
        "reason": "",
    }

    if binary_ratio < binary_ratio_thr:
        info["reason"] = f"alpha 非二值（二值占比 {binary_ratio:.1%} < {binary_ratio_thr:.0%}），判定为未完成抠图"
        return False, info
    if border_transparent < border_transparent_thr:
        info["reason"] = f"边框未透明（{border_transparent:.1%} < {border_transparent_thr:.0%}），背景未被掏空"
        return False, info
    if transparent_ratio >= 0.999:
        info["reason"] = "整图几乎全透明，无可沿用主体"
        return False, info

    info["reason"] = "检测到干净且完整的透明通道，直接沿用"
    return True, info


def default_output_path(input_path: str, output_dir: Optional[str] = None,
                        suffix: str = "_nobg") -> str:
    """根据输入路径推导默认输出路径：<dir>/<stem><suffix>.png。"""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(input_path))
    return os.path.join(out_dir, f"{stem}{suffix}.png")
