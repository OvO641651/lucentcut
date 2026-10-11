"""诊断脚本：对指定目录所有图片跑 颜色引擎 / AI 引擎，输出对比与统计。

作者：LucentCut 调试工具（非交付代码）
用法：
    .venv/Scripts/python.exe tools/diagnose.py "image/动漫表情包图" -o output_diag
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from src.core import ai_segment, colorkey, io, refine

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def checkerboard(h, w, size=12, c1=(200, 200, 200), c2=(255, 255, 255)):
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // size + xx // size) % 2).astype(bool)
    board = np.zeros((h, w, 3), np.uint8)
    board[m] = c1
    board[~m] = c2
    return board


def compose(rgb, alpha, bg_mode="checker"):
    h, w = alpha.shape
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    if bg_mode == "checker":
        bg = checkerboard(h, w)
    elif bg_mode == "white":
        bg = np.full((h, w, 3), 255, np.uint8)
    else:
        bg = np.zeros((h, w, 3), np.uint8)
    out = rgb.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)
    return out.astype(np.uint8)


def thumb(arr, max_side=420):
    h, w = arr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return arr
    im = Image.fromarray(arr)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("-o", "--outdir", default="output_diag")
    ap.add_argument("--no-ai", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = [n for n in sorted(os.listdir(args.indir))
             if os.path.splitext(n)[1].lower() in IMG_EXT]
    print(f"共 {len(names)} 张，AI 可用={ai_segment.is_available()}\n")
    print(f"{'文件':<26}{'尺寸':<12}{'mode':<9}{'二值':<7}{'rgba':<6}"
          f"{'颜色保留':<9}{'AI保留':<9}耗时")
    print("-" * 100)

    rows = []
    for n in names:
        p = os.path.join(args.indir, n)
        try:
            rgb, oa = io.load_image(p)
        except Exception as e:
            print(f"{n:<26} 读取失败 {e}")
            continue
        h, w = rgb.shape[:2]
        mode = colorkey.detect_mode(rgb)
        has_alpha = oa is not None
        # 颜色引擎
        t0 = time.time()
        try:
            ca, cmode, cbg = colorkey.remove_solid_bg(rgb, mode="auto")
            crgb, ca = refine.refine(rgb, ca, cbg, feather_radius=pipeline_feather(h, w),
                                     despill_on=True)
            color_ok = True
        except Exception as e:
            ca, cmode, color_ok = None, f"err({e})", False
        tc = time.time() - t0
        # AI 引擎
        ai_keep = "-"
        ai_alpha = None
        ta = 0.0
        if ai_segment.is_available() and not args.no_ai:
            t0 = time.time()
            try:
                r = ai_segment.remove_background(rgb)
                ai_alpha = refine.harden_alpha(r.alpha)
                ai_keep = f"{float(np.mean(ai_alpha >= 128)):.1%}"
            except Exception as e:
                ai_keep = f"err"
                traceback.print_exc()
            ta = time.time() - t0

        ckeep = f"{float(np.mean(ca >= 128)):.1%}" if color_ok else "-"
        print(f"{n:<26}{w}x{h:<8}{mode:<9}{('Y' if has_alpha else 'N'):<7}"
              f"{('Y' if has_alpha else 'N'):<6}{ckeep:<9}{ai_keep:<9}"
              f"{tc:.1f}/{ta:.1f}s")

        # 保存三栏对比：原图 / 颜色 / AI
        panels = [thumb(rgb)]
        labels = ["orig"]
        if color_ok:
            panels.append(thumb(compose(crgb, ca)))
            labels.append("color")
        if ai_alpha is not None:
            panels.append(thumb(compose(rgb, ai_alpha)))
            labels.append("ai")
        H = max(p.shape[0] for p in panels)
        W = sum(p.shape[1] for p in panels)
        canvas = np.full((H, W, 3), 245, np.uint8)
        x = 0
        for p in panels:
            canvas[:p.shape[0], x:x + p.shape[1]] = p
            x += p.shape[1]
        stem = os.path.splitext(n)[0]
        Image.fromarray(canvas).save(os.path.join(args.outdir, f"{stem}_cmp.png"))
        rows.append((n, cmode, ckeep, ai_keep))

    print("\n对比图已保存到", args.outdir)


def pipeline_feather(h, w):
    from src.core.pipeline import auto_feather_radius
    return auto_feather_radius(h, w)


if __name__ == "__main__":
    main()
