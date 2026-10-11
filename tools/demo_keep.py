"""演示：白底图上的白裙子被颜色键控抠掉后，用「框内保留」把它救回来。"""
import os, sys
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import io, pipeline, interactive

SRC = "image/动漫表情包图/doro1.png"
OUT = "output/selftest/_框内保留演示.png"

rgb, _ = io.load_image(SRC)
h, w = rgb.shape[:2]

# 1) 纯颜色键控：白裙会被一起抠掉
_, a_color, *_ = pipeline.process_array(rgb, None, engine="color")
# 2) 找出"AI 认为主体、却被颜色键控删掉"的最大连通块 —— 这就是被误删的裙子
from src.core import ai_segment
ai = ai_segment.predict_alpha(rgb)
lost = (ai >= 170) & (a_color < 128)
num, labels, stats, _ = cv2.connectedComponentsWithStats(lost.astype(np.uint8), 8)
keep = lost.astype(np.uint8) * 255   # 三条裙子（多个互不相连的区域）一起保留
print("被误删区域像素数:", int((keep > 0).sum()))

# 3) 模拟用户沿裙子描一圈 → 框内保留
_, a_keep = interactive.apply_override(rgb, a_color, rgb, keep, None, feather_px=1)

# 4) 三栏对比图
def panel(rgb, alpha, title_h=0):
    rgba = np.dstack([rgb, alpha])
    im = Image.fromarray(rgba, "RGBA")
    base = Image.new("RGBA", (w, h), (230, 30, 40, 255))
    base.paste(im, (0, 0), im)
    return np.asarray(base.convert("RGB"))

p1 = panel(rgb, np.full((h, w), 255, np.uint8))
p2 = panel(rgb, a_color)
p3 = panel(rgb, a_keep)
canvas = np.concatenate([p1, p2, p3], axis=1)
Image.fromarray(canvas).save(OUT)
print("颜色引擎主体保留率: %.1f%%" % (100 * np.mean(a_color >= 128)))
print("框内保留后保留率  : %.1f%%" % (100 * np.mean(a_keep >= 128)))
print("已保存:", OUT)
