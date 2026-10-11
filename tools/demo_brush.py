"""演示 M10：点选笔刷修掉「白色细节被抠掉」。

场景对应用户反馈：白色眼睛高光 / 白裙子被算法当成白背景删掉了。
这里模拟用户用「点选」在那些位置涂几笔绿色（作用=保留），看结果是否恢复。

每行：左=原图 / 中=纯颜色键控（白细节被删）/ 右=点选笔刷保留后
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import ai_segment, interactive, io, pipeline  # noqa: E402

CASES = [
    ("image/动漫表情包图/deepseek2.png", "color"),   # 眼睛里的白色高光
    ("image/动漫表情包图/doro1.png", "color"),       # 白色裙子
]
OUT = "output/selftest/_点选笔刷演示.png"
RED = (230, 30, 40, 255)          # 红色底板，便于看透明区


def panel(rgb, alpha, w, h):
    im = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    base = Image.new("RGBA", (w, h), RED)
    base.paste(im, (0, 0), im)
    return np.asarray(base.convert("RGB"))


def scribble_points(mask: np.ndarray, radius: int):
    """在 mask 内按笔刷间距撒点，模拟用户拿笔刷把这一片刷一遍。"""
    step = max(3, int(radius) // 2)
    pts = []
    ys, xs = np.nonzero(mask)
    for y in range(0, mask.shape[0], step):
        row = mask[y]
        for x in range(0, mask.shape[1], step):
            if row[x]:
                pts.append((x, y))
    return pts


def one_row(path, engine):
    rgb, _ = io.load_image(path)
    h, w = rgb.shape[:2]

    _, a0, mode, *_ = pipeline.process_array(rgb, None, engine=engine)

    # 找出「原本亮白、却被打成透明」的区域 —— 模拟用户会去涂的地方
    ai = ai_segment.predict_alpha(rgb)
    bright = rgb.min(axis=2) > 225
    lost = bright & (a0 < 128) & (ai >= 150)

    radius = 12
    pts = scribble_points(lost, radius)
    keep = interactive.points_to_mask(rgb.shape, pts, radius)
    rgb_k, a_k = interactive.apply_override(rgb, a0, rgb, keep, None, feather_px=1)

    name = os.path.basename(path)
    if lost.any():
        print(f"{name} [{engine}]: 白色细节被删 {int(lost.sum())} px，"
              f"涂 {len(pts)} 笔后 alpha 均值 {float(a0[lost].mean()):.1f} → "
              f"{float(a_k[lost].mean()):.1f}，仍被删 {int((a_k[lost] < 128).sum())} px")
    else:
        print(f"{name} [{engine}]: 无白色细节被删")

    return np.concatenate([
        panel(rgb, np.full((h, w), 255, np.uint8), w, h),
        panel(rgb, a0, w, h),
        panel(rgb_k, a_k, w, h),
    ], axis=1)


def main():
    for path, engine in CASES:
        img = one_row(path, engine)
        tag = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join("output/selftest", f"_点选笔刷演示_{tag}.png")
        Image.fromarray(img).save(out)
        print("已保存:", out)


if __name__ == "__main__":
    main()
