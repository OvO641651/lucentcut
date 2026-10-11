"""评估脚本：对目录跑 pipeline 各引擎，输出对比图与统计。

用法：
  .venv/Scripts/python.exe tools/eval_out.py "image/动漫表情包图" -o output_eval --engines hybrid color ai
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from src.core import io, pipeline
from tools.diagnose import compose, thumb

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("-o", "--outdir", default="output_eval")
    ap.add_argument("--engines", nargs="*", default=["hybrid"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    names = [n for n in sorted(os.listdir(args.indir))
             if os.path.splitext(n)[1].lower() in IMG_EXT]
    print(f"共 {len(names)} 张，引擎 {args.engines}")
    hdr = f"{'文件':<30}{'尺寸':<12}" + "".join(f"{e+'(保留)':<16}" for e in args.engines) + "耗时"
    print(hdr)
    print("-" * 110)

    for n in names:
        p = os.path.join(args.indir, n)
        try:
            rgb, oa = io.load_image(p)
        except Exception as e:
            print(f"{n:<30} 读取失败 {e}")
            continue
        h, w = rgb.shape[:2]
        panels = [thumb(rgb)]
        info = []
        t0 = time.time()
        for eng in args.engines:
            try:
                ro, ao, mode, rescued, reused, warns = pipeline.process_array(
                    rgb, oa, engine=eng)
                panels.append(thumb(compose(ro, ao)))
                info.append(f"{mode}({np.mean(ao >= 128):.0%})")
            except Exception as e:
                info.append(f"ERR:{type(e).__name__}")
        dt = time.time() - t0
        print(f"{n:<30}{w}x{h:<8}" +
              "".join(f"{s:<16}" for s in info) + f"{dt:.1f}s")

        H = max(pp.shape[0] for pp in panels)
        W = sum(pp.shape[1] for pp in panels)
        canvas = np.full((H, W, 3), 245, np.uint8)
        x = 0
        for pp in panels:
            canvas[:pp.shape[0], x:x + pp.shape[1]] = pp
            x += pp.shape[1]
        Image.fromarray(canvas).save(
            os.path.join(args.outdir, os.path.splitext(n)[0] + ".png"))

    print("\n对比图 ->", args.outdir)


if __name__ == "__main__":
    main()
