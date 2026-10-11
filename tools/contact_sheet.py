"""生成"原图 | 结果"对比总览大图，便于快速肉眼检查。

用法：.venv/Scripts/python.exe tools/contact_sheet.py "image/X" -o output/总览.png -r out_dir
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core import io
from tools.diagnose import compose, thumb

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_font(size=16):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("-o", "--out", default="output/contact_sheet.png")
    ap.add_argument("-r", "--resultdir", required=True)
    ap.add_argument("--cell", type=int, default=280)
    args = ap.parse_args()

    names = [n for n in sorted(os.listdir(args.indir))
             if os.path.splitext(n)[1].lower() in IMG_EXT]
    cell = args.cell
    pad = 10
    label_h = 22
    cols = 3
    rows = (len(names) + cols - 1) // cols
    cw = cell * 2 + pad * 3
    ch = cell + label_h + pad * 2
    W = cols * cw
    H = rows * ch
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)

    def fit(arr, side):
        im = Image.fromarray(arr)
        im.thumbnail((side, side), Image.LANCZOS)
        return im

    for i, n in enumerate(names):
        r, c = divmod(i, cols)
        x0 = c * cw
        y0 = r * ch
        try:
            rgb, _ = io.load_image(os.path.join(args.indir, n))
        except Exception:
            continue
        out_path = os.path.join(args.resultdir,
                                os.path.splitext(n)[0] + "_nobg.png")
        draw.text((x0 + pad, y0 + 4), n, fill=(20, 20, 20), font=font)
        src = fit(rgb, cell)
        canvas.paste(src, (x0 + pad, y0 + label_h + pad))
        if os.path.isfile(out_path):
            res_rgb, res_a = io.load_image(out_path)
            comp = compose(res_rgb, res_a if res_a is not None else np.full(res_rgb.shape[:2], 255, np.uint8))
            res = fit(comp, cell)
            canvas.paste(res, (x0 + pad * 2 + cell, y0 + label_h + pad))
        else:
            draw.text((x0 + pad * 2 + cell, y0 + label_h + pad), "缺失结果",
                      fill=(200, 0, 0), font=font)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    print("总览图 ->", args.out, canvas.size)


if __name__ == "__main__":
    main()
