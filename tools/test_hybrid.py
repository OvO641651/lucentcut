"""试验：颜色引擎 + AI 补齐 的混合方案效果对比。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from src.core import ai_segment, colorkey, io, refine
from tools.diagnose import compose, thumb

IMAGE = "image/动漫表情包图"
OUT = "output_hybrid"
NAMES = ["deepseek.jpg", "deepseek1.jpg", "deepseek2.png", "deekseep3.jpg",
         "doro1.png", "8F9E75B2B7142651DB62D37D1B5FF80D.jpg",
         "0EC868CDA89CCB1F49310AAFFAD6D747.jpg",
         "FEA740A6443CA7D87F32E306BE8D3D8D.jpg",
         "E3F8226F2750FC4ED5E2C159D4ECE4A7.jpg"]


def keep_large(mask, min_ratio=0.0015):
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return mask
    thr = max(64, int(mask.size * min_ratio))
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            out[lab == i] = 1
    return out.astype(bool)


def hybrid(rgb, color_alpha, ai_alpha, hi=180, lo=120):
    subj = keep_large(ai_alpha >= hi)
    restore = subj & (color_alpha < lo)
    out = color_alpha.copy()
    out[restore] = 255
    return out, restore


def main():
    os.makedirs(OUT, exist_ok=True)
    for n in NAMES:
        p = os.path.join(IMAGE, n)
        if not os.path.isfile(p):
            print("skip", n)
            continue
        rgb, oa = io.load_image(p)
        h, w = rgb.shape[:2]
        # color
        try:
            ca, mode, bg = colorkey.remove_solid_bg(rgb, mode="auto")
            crgb, ca2 = refine.refine(rgb, ca, bg, feather_radius=1, despill_on=True)
        except Exception as e:
            ca2, crgb, bg = None, None, None
        # ai
        ai = ai_segment.remove_background(rgb)
        aia = ai.alpha
        if ca2 is None:
            panels = [rgb, compose(rgb, aia)]
            labels = "orig|ai"
        else:
            hy, rest = hybrid(rgb, ca2, aia)
            panels = [rgb, compose(crgb, ca2), compose(rgb, aia), compose(rgb, hy)]
            labels = f"orig|color|ai|hybrid restore={float(rest.mean()):.2%}"
        panels = [thumb(x) for x in panels]
        H = max(p.shape[0] for p in panels)
        W = sum(p.shape[1] for p in panels)
        canvas = np.full((H, W, 3), 245, np.uint8)
        x = 0
        for pp in panels:
            canvas[:pp.shape[0], x:x + pp.shape[1]] = pp
            x += pp.shape[1]
        stem = os.path.splitext(n)[0]
        Image.fromarray(canvas).save(os.path.join(OUT, f"{stem}.png"))
        print(f"{n:<40} {labels}")


if __name__ == "__main__":
    main()
