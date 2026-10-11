"""生成 旧(u2netp) vs 新(M12 matting) 抠图前后对比图与 HTML 画廊。

用法：
    .venv/Scripts/python.exe tools/make_comparison.py

输出目录：image/compare_output/
    - <name>_old.png   旧引擎(u2netp, 320x320) 棋盘底合成
    - <name>_new.png   新引擎(M12 matting, BiRefNet) 棋盘底合成
    - <name>_ideal.png 参考"其他软件"理想结果（若有）
    - index.html       并排对比画廊（含原图、旧、新、理想）
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core import pipeline  # noqa: E402

# 棋盘底（灰度方格），用于直观显示透明区
CHECK_A = (128, 128, 128)
CHECK_B = (200, 200, 200)
CHECK = 16


def checker(h: int, w: int) -> np.ndarray:
    out = np.empty((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            out[y, x] = CHECK_A if ((x // CHECK + y // CHECK) % 2 == 0) else CHECK_B
    return out


def composite(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """把 RGBA 合成到棋盘底上。"""
    h, w = rgb.shape[:2]
    bg = checker(h, w).astype(np.float32)
    fg = rgb.astype(np.float32)
    a = alpha.astype(np.float32) / 255.0
    a3 = a[..., None]
    comp = fg * a3 + bg * (1 - a3)
    return comp.clip(0, 255).astype(np.uint8)


def cut(path: str, engine: str, reuse: bool):
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    t0 = time.time()
    rgb_out, alpha, mode, *_ = pipeline.process_array(
        rgb, None, engine=engine, reuse_existing_alpha=reuse)
    dt = time.time() - t0
    return composite(rgb_out, alpha), mode, dt


def main() -> int:
    out_dir = os.path.join(ROOT, "image", "compare_output")
    os.makedirs(out_dir, exist_ok=True)

    cases = [
        {
            "name": "1_动漫四人合影",
            "src": os.path.join(ROOT, "image", "1.jpg"),
            "ideal": os.path.join(ROOT, "image", "2.png"),
        },
        {
            "name": "3D_4aef3a_复杂背景",
            "src": os.path.join(ROOT, "image", "3D角色人物图",
                                "4aef3a4ca13afa55bd7a5d4a385fc63d_0.jpeg"),
            "ideal": "",
        },
        {
            "name": "3D_2c05",
            "src": os.path.join(ROOT, "image", "3D角色人物图",
                                "2c05beadb26d95e25979cc0c45e93a37_720.jpeg"),
            "ideal": os.path.join(ROOT, "image", "3D角色人物图", "output",
                                  "2c05beadb26d95e25979cc0c45e93a37_720.png"),
        },
    ]

    rows = []
    for c in cases:
        name = c["name"]
        print(f"\n=== {name} ===")
        if not os.path.isfile(c["src"]):
            print(f"  跳过（源图不存在）：{c['src']}")
            continue

        # 原图（另存一份到输出目录，保证画廊用相对路径可离线查看）
        _orig = Image.open(c["src"]).convert("RGB")
        src_path = os.path.join(out_dir, f"{name}_src.png")
        _orig.save(src_path)

        # 旧引擎
        old_img, old_mode, old_dt = cut(c["src"], "ai", reuse=False)
        old_path = os.path.join(out_dir, f"{name}_old.png")
        Image.fromarray(old_img).save(old_path)
        print(f"  旧(u2netp)  mode={old_mode}  {old_dt:.1f}s  -> {os.path.basename(old_path)}")

        # 新引擎
        new_img, new_mode, new_dt = cut(c["src"], "matting", reuse=False)
        new_path = os.path.join(out_dir, f"{name}_new.png")
        Image.fromarray(new_img).save(new_path)
        print(f"  新(matting) mode={new_mode}  {new_dt:.1f}s -> {os.path.basename(new_path)}")

        # 理想参考
        ideal_path = ""
        if c.get("ideal") and os.path.isfile(c["ideal"]):
            ideal = Image.open(c["ideal"]).convert("RGBA")
            ideal_arr = np.asarray(ideal)
            ideal_comp = composite(ideal_arr[..., :3], ideal_arr[..., 3])
            ideal_path = os.path.join(out_dir, f"{name}_ideal.png")
            Image.fromarray(ideal_comp).save(ideal_path)
            print(f"  理想参考 -> {os.path.basename(ideal_path)}")

        rows.append({
            "name": name,
            "src": src_path,
            "old": old_path,
            "new": new_path,
            "ideal": ideal_path,
            "old_dt": old_dt,
            "new_dt": new_dt,
        })

    # 生成 HTML 画廊
    html = ["<html><head><meta charset='utf-8'>",
            "<title>LucentCut 抠图对比：旧(u2netp) vs 新(M12 matting)</title>",
            "<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;"
            "background:#1e1e1e;color:#ddd;margin:0;padding:24px;}"
            "h1{font-size:20px;} .case{margin:24px 0;}"
            "table{border-collapse:collapse;margin-top:8px;}"
            "td,th{border:1px solid #444;padding:4px 8px;text-align:center;}"
            "th{background:#333;font-weight:600;}"
            "img{max-height:360px;border:1px solid #555;}"
            ".tag{display:inline-block;padding:2px 8px;border-radius:10px;"
            "font-size:12px;margin-left:6px;}"
            ".old{background:#5a3a3a;}.new{background:#2e5a3a;}"
            ".ideal{background:#3a4a6a;}</style></head><body>",
            "<h1>LucentCut 抠图质量对比</h1>",
            "<p>棋盘底用于显示透明区域。<b>旧</b>=内置 u2netp(320×320)，"
            "<b>新</b>=M12 精细抠图引擎(BiRefNet 1024×1024)，"
            "<b>理想</b>=其他软件处理结果。</p>"]

    for r in rows:
        html.append(f"<div class='case'><h2>{r['name']}</h2>")
        html.append("<table><tr>")
        html.append("<th>原图</th><th>旧 u2netp</th><th>新 M12</th>")
        if r["ideal"]:
            html.append("<th>理想(其他软件)</th>")
        html.append("</tr><tr>")
        html.append(f"<td><img src='{os.path.basename(r['src'])}'></td>")
        html.append(f"<td><img src='{os.path.basename(r['old'])}'>"
                    f"<br><span class='tag old'>{r['old_dt']:.1f}s</span></td>")
        html.append(f"<td><img src='{os.path.basename(r['new'])}'>"
                    f"<br><span class='tag new'>{r['new_dt']:.1f}s</span></td>")
        if r["ideal"]:
            html.append(f"<td><img src='{os.path.basename(r['ideal'])}'>"
                        f"<br><span class='tag ideal'>参考</span></td>")
        html.append("</tr></table></div>")

    html.append("</body></html>")
    idx = os.path.join(out_dir, "index.html")
    with open(idx, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"\n画廊已生成：{idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
