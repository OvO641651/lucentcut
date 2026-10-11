"""M1 验证脚本：跑通 image/ 下所有样本，输出透明 PNG 并生成对比图。

用法：
    python tests/run_samples.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core import pipeline  # noqa: E402

IMG_DIR = os.path.join(ROOT, "image")
OUT_DIR = os.path.join(ROOT, "output")


def checkerboard(w: int, h: int, cell: int = 12) -> Image.Image:
    """生成灰白棋盘格背景，便于观察透明区域。"""
    board = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    light, dark = (235, 235, 235), (200, 200, 200)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (x // cell + y // cell) % 2 == 0:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=light)
            else:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=dark)
    return board


def panel(img: Image.Image, size: int) -> Image.Image:
    """等比缩放到最长边 size。"""
    im = img.copy()
    im.thumbnail((size, size), Image.LANCZOS)
    return im


def composite(img: Image.Image, color) -> Image.Image:
    """把 RGBA 图叠加到指定纯色背景上（用于暴露白边）。"""
    bg = Image.new("RGB", img.size, color)
    bg.paste(img, (0, 0), img)
    return bg


def halo_metric(rgba: Image.Image) -> float:
    """边缘像素平均亮度：白底图上该值应较低（说明没残留白边）。"""
    arr = np.asarray(rgba, dtype=np.uint8)
    a = arr[:, :, 3]
    edge = (a > 20) & (a < 235)
    if edge.sum() == 0:
        return 0.0
    return float(arr[:, :, :3][edge].mean())


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    names = sorted(
        f for f in os.listdir(IMG_DIR)
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )

    rows = []
    print(f"{'文件':<20}{'模式':<9}{'尺寸':<14}{'背景透明率':<12}{'主体保留率':<12}{'边缘亮度':<10}{'告警'}")
    print("-" * 100)

    for name in names:
        src = os.path.join(IMG_DIR, name)
        out = os.path.join(OUT_DIR, os.path.splitext(name)[0] + "_nobg.png")
        try:
            res = pipeline.process_image(src, out)
        except ValueError as exc:
            print(f"{name:<20}{'跳过':<9}{'-':<14}{'-':<12}{'-':<12}{'-':<10}{exc}")
            rows.append((name, Image.open(src).convert("RGB"), None, None, "复杂背景(待二期)"))
            continue

        result_img = Image.open(out).convert("RGBA")
        bg = checkerboard(*result_img.size)
        bg.paste(result_img, (0, 0), result_img)
        on_magenta = composite(result_img, (255, 0, 255))

        halo = halo_metric(result_img)
        warn = "；".join(res.warnings) if res.warnings else "OK"
        print(f"{name:<20}{res.mode:<9}{res.width}x{res.height:<8}"
              f"{res.bg_transparent_ratio:<12.1%}{res.fg_keep_ratio:<12.1%}"
              f"{halo:<10.1f}{warn}")
        rows.append((name, Image.open(src).convert("RGB"), bg, on_magenta, warn))

    # 生成对比图：原图 | 棋盘格结果 | 洋红底结果
    cell_w, cell_h = 440, 380
    cols = 3
    n_rows = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGB", (cell_w * cols, cell_h * n_rows), (245, 245, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()

    for idx, (name, before, after, on_magenta, note) in enumerate(rows):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h
        draw.text((x0 + 8, y0 + 6), name, fill=(20, 20, 20), font=font)
        draw.text((x0 + 8, y0 + 26), "原图", fill=(120, 120, 120), font=font)
        draw.text((x0 + 150, y0 + 26), "结果(棋盘格)", fill=(120, 120, 120), font=font)
        draw.text((x0 + 292, y0 + 26), "结果(洋红底)", fill=(120, 120, 120), font=font)
        canvas.paste(panel(before, 134), (x0 + 8, y0 + 46))
        if after is not None:
            canvas.paste(panel(after, 134), (x0 + 150, y0 + 46))
            canvas.paste(panel(on_magenta, 134), (x0 + 292, y0 + 46))
        draw.text((x0 + 8, y0 + 196), note[:60], fill=(90, 90, 90), font=font)

    preview = os.path.join(OUT_DIR, "_对比预览.png")
    canvas.save(preview)
    print("-" * 90)
    print(f"结果输出目录: {OUT_DIR}")
    print(f"对比预览图  : {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
