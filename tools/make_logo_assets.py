"""LucentCut LOGO 品牌资产生成脚本。

输入：assets/logo/ 下 AI 生成的原始 icon（白底 RGB，lucentcut_icon_source.png）。
输出（均落在 assets/logo/）：
  - lucentcut_icon_transparent.png   透明底主图标（1024）
  - lucentcut_icon_{512..16}.png     各尺寸 PNG
  - lucentcut.ico                    多尺寸 exe 图标（16~256）
  - lucentcut_lockup_light.png       横向组合标（浅色底用，彩色字）
  - lucentcut_lockup_dark.png        横向组合标（深色底用，白字）
  - lucentcut_banner.png             深色横幅（深底用，含标语）

方法：把「与画布边框连通的近白像素」置为透明（cv2.floodFill），
图标内部的亮色（照片卡片）是封闭区域，不受影响；再 erode 1px + 轻微模糊
消除白边并做抗锯齿。文字用 Segoe UI Bold / 微软雅黑 Bold 渲染，绝不乱码。
"""
from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO = os.path.join(ROOT, "assets", "logo")
SRC_ICON = os.path.join(LOGO, "lucentcut_icon_source.png")

FONT_EN_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CN_BOLD = "C:/Windows/Fonts/msyhbd.ttc"

C_CYAN = (14, 165, 233)     # #0EA5E9
C_VIOLET = (124, 58, 237)   # #7C3AED
C_SLATE = (30, 41, 59)      # #1E293B


# ---------------------------------------------------------------- icon alpha
def strip_border_white(rgb: np.ndarray, thresh: int = 242) -> np.ndarray:
    """把与画布四条边框连通的近白像素设为透明，返回 alpha(0-255)。"""
    h, w = rgb.shape[:2]
    white = (rgb >= thresh).all(axis=2).astype(np.uint8)
    ff = white.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for sx, sy in seeds:
        if ff[sy, sx] == 1:                       # 只在真·白角上开洪泛
            cv2.floodFill(ff, mask, (sx, sy), 2)
    alpha = np.where(ff == 2, 0, 255).astype(np.uint8)
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)  # 去白边
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)                      # 抗锯齿
    return alpha


def build_icon() -> Image.Image:
    im = Image.open(SRC_ICON).convert("RGB")
    arr = np.asarray(im)
    alpha = strip_border_white(arr)
    rgba = np.dstack([arr, alpha])
    out = Image.fromarray(rgba, "RGBA")
    a = np.asarray(out)[:, :, 3]
    ys, xs = np.where(a > 8)
    out = out.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    S = 1024
    out = out.resize((S, S), Image.LANCZOS)
    out.save(os.path.join(LOGO, "lucentcut_icon_transparent.png"))
    return out


# ---------------------------------------------------------------- text utils
def gradient_text(text: str, font: ImageFont.FreeTypeFont,
                  c1=C_CYAN, c2=C_VIOLET) -> Image.Image:
    """渲染渐变文字，返回 RGBA 图（带 alpha 蒙版）。"""
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    bb = probe.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 10
    m = Image.new("L", (tw + 2 * pad, th + 2 * pad), 0)
    ImageDraw.Draw(m).text((pad - bb[0], pad - bb[1]), text, font=font, fill=255)
    cols = np.zeros((m.height, m.width, 3), np.uint8)
    for x in range(m.width):
        t = x / max(1, m.width - 1)
        cols[:, x] = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    out.paste(Image.fromarray(cols, "RGB"), (0, 0), m)
    return out


def solid_text(text: str, font: ImageFont.FreeTypeFont, color) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    bb = probe.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 10
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - bb[0], pad - bb[1]), text, font=font, fill=color)
    return img


# ---------------------------------------------------------------- lockups
def build_lockup(icon: Image.Image, dark: bool, tagline: bool = True,
                 out_name: str = "", banner: bool = False):
    f_main = ImageFont.truetype(FONT_EN_BOLD, 300)
    f_tag = ImageFont.truetype(FONT_CN_BOLD, 78)

    if banner:
        em_size = 620
        word = solid_text("LucentCut", f_main, (255, 255, 255))
    else:
        em_size = 470
        word = (solid_text("LucentCut", f_main, (255, 255, 255)) if dark
                else gradient_text("LucentCut", f_main))

    tag = None
    if tagline:
        tag_color = (203, 213, 225) if (dark or banner) else (100, 116, 139)
        tag = solid_text("AI 智能抠图 · 一键去除背景", f_tag, tag_color)

    em = icon.resize((em_size, em_size), Image.LANCZOS)
    gap = 70 if not banner else 90
    pad = 70 if not banner else 110

    tw = word.width
    tt = tag.width if tag else 0
    text_w = max(tw, tt)
    core_h = max(em_size, word.height + (30 + tag.height if tag else 0))

    W = pad * 2 + em_size + gap + text_w
    H = pad * 2 + core_h
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0) if not banner else (15, 23, 42, 255))

    cy = H // 2
    canvas.alpha_composite(em, (pad, cy - em_size // 2))

    tx = pad + em_size + gap
    if tag:
        block_h = word.height + 30 + tag.height
        wy = cy - block_h // 2
        canvas.alpha_composite(word, (tx, wy))
        canvas.alpha_composite(tag, (tx, wy + word.height + 30))
    else:
        canvas.alpha_composite(word, (tx, cy - word.height // 2))

    path = os.path.join(LOGO, out_name)
    canvas.save(path)
    return path, canvas.size


def main():
    icon = build_icon()
    print("[icon] transparent saved:", icon.size)

    for s in (512, 256, 128, 64, 48, 32, 16):
        icon.resize((s, s), Image.LANCZOS).save(
            os.path.join(LOGO, f"lucentcut_icon_{s}.png"))
    ico = os.path.join(LOGO, "lucentcut.ico")
    icon.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                          (64, 64), (128, 128), (256, 256)])
    print("[icon] ico saved:", ico)

    p1 = build_lockup(icon, dark=False, tagline=True,
                      out_name="lucentcut_lockup_light.png")
    p2 = build_lockup(icon, dark=True, tagline=True,
                      out_name="lucentcut_lockup_dark.png")
    p3 = build_lockup(icon, dark=True, tagline=True, banner=True,
                      out_name="lucentcut_banner.png")
    for p, sz in (p1, p2, p3):
        print("[lockup]", os.path.basename(p), sz)


if __name__ == "__main__":
    main()
