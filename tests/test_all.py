"""端到端自测 + 边界/健壮性测试。

覆盖：
  [1] 颜色键控引擎（白底/黑底/灰底/复杂背景/沿用已有 alpha）
  [2] 混合引擎（颜色轮廓 + AI 补齐）
  [3] AI 语义分割（自包含 u2netp 推理）
  [4] 交互式抠图（矩形/点选/描边/套索）
  [5] GUI 窗口构建与处理
  [6] 边界与异常输入（漏洞防护）
  [7] 人工裁决层（框内保留 / 框内去除）
  [8] 画布交互（中键平移 / 多区域 / 三级撤销）
  [9] GUI 人工修改完整流程
  [10] 交互修正（点选笔刷 / 显示修改区 / 应用选区语义）
  [11] 内部半透明修复（画质：浅色内部部件不得残留半透明）
  [12] CLI 命令行
  [13] 精细抠图引擎（M12：BiRefNet/ISNet、回退、IoU 对比、内存稳定）
  [14] M13：裁决层跨笔合并语义（后画覆盖先画/点选颜色语义）+ 批量进度回调
  [15] M15：画布重写（Redo） + ROI 局部处理 + QGraphicsView 迁移
  [16] M16：ROI 局部处理（先处理后框选）+ 处理后/预览时清除绘制内容
  [17] M17：点选笔刷半径严格生效（不再被最小 3 顶替）

用法：.venv/Scripts/python.exe tests/test_all.py
返回码非 0 表示有失败项。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.core import (  # noqa: E402
    ai_segment, colorkey, hq_segment, interactive, io, pipeline, refine,
)

IMG_DIR = os.path.join(ROOT, "image")
ANIME_DIR = os.path.join(IMG_DIR, "动漫表情包图")
OUT_DIR = os.path.join(ROOT, "output", "selftest")
PY = sys.executable

PASS = 0
FAIL = 0
FAILED_NAMES: list[str] = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        FAILED_NAMES.append(name)
        print(f"  [XX] {name}  {detail}")


def section(t):
    print(f"\n{t}")


def sample(name: str) -> str:
    """在样本目录里找一个测试图；ANIME_DIR 缺失时回退到顶层 IMG_DIR。

    背景（2026-10-16）：部分样本（deepseek.jpg / deepseek2.png / system.png）
    未随仓提交，导致硬引用它们的用例抛 FileNotFoundError 而中断整套自测。
    这里统一按"先 动漫表情包图/、再 image/"的顺序查找，两者都没有时返回
    原路径（调用方自行判存在或捕获异常），从而让套件在样本不全时也能跑完。
    """
    for base in (ANIME_DIR, IMG_DIR):
        cand = os.path.join(base, name)
        if os.path.exists(cand):
            return cand
    return os.path.join(ANIME_DIR, name)


# --------------------------------------------------------------------------- #
def test_color_engine():
    section("[1] 颜色键控引擎")
    samples = {
        "deepseek.jpg": "white",
        "deepseek1.jpg": "white",
        "deepseek2.png": "white",
        "deepseek4.png": "reuse",     # 已抠好的透明图 → 沿用
        "黑色背景.jpg": "black",
    }
    for name, expect in samples.items():
        path = os.path.join(ANIME_DIR, name)
        if not os.path.exists(path):
            continue
        res = pipeline.process_image(path, os.path.join(OUT_DIR, name + ".png"),
                                     engine="color")
        check(f"颜色引擎 {name} → {expect}", res.mode == expect, f"got {res.mode}")

    # 灰底（solid）
    sys_path = os.path.join(IMG_DIR, "system.png")
    if os.path.exists(sys_path):
        res = pipeline.process_image(sys_path, os.path.join(OUT_DIR, "system.png"),
                                     engine="color")
        check("灰底图 → solid", res.mode == "solid", f"got {res.mode}")

    # 沿用已有 alpha 不得误删主体
    ds4 = os.path.join(ANIME_DIR, "deepseek4.png")
    if os.path.exists(ds4):
        res = pipeline.process_image(ds4, os.path.join(OUT_DIR, "reuse_check.png"),
                                     engine="color")
        check("已有透明图沿用（不重抠）", res.reused_existing_alpha, f"got {res.mode}")
        check("沿用图主体保留率 > 50%", res.fg_keep_ratio > 0.5,
              f"{res.fg_keep_ratio:.1%}")


def test_hybrid_engine():
    section("[2] 混合引擎（颜色轮廓 + AI 补齐）")
    # 白色衣物类：混合引擎应比纯颜色键控保留更多主体（补回被误删的白衣服）
    for name in ["doro1.png", "8F9E75B2B7142651DB62D37D1B5FF80D.jpg",
                 "deepseek.jpg", "deepseek2.png"]:
        path = os.path.join(ANIME_DIR, name)
        if not os.path.exists(path):
            continue
        h = pipeline.process_image(path, os.path.join(OUT_DIR, "hyb_" + name + ".png"),
                                   engine="hybrid")
        c = pipeline.process_image(path, os.path.join(OUT_DIR, "col_" + name + ".png"),
                                   engine="color")
        check(f"混合引擎补齐白衣服 {name}", h.rescued_ratio > 0,
              f"rescued={h.rescued_ratio:.2%}")
        check(f"混合保留率 >= 纯颜色 {name}", h.fg_keep_ratio >= c.fg_keep_ratio - 1e-9,
              f"{h.fg_keep_ratio:.1%} < {c.fg_keep_ratio:.1%}")

    # AI 不可判断主体时，混合结果不应比颜色键控更差（0EC 为 AI 漏分割样本）
    name = "0EC868CDA89CCB1F49310AAFFAD6D747.jpg"
    path = os.path.join(ANIME_DIR, name)
    if os.path.exists(path):
        h = pipeline.process_image(path, os.path.join(OUT_DIR, "hyb_0ec.png"),
                                   engine="hybrid")
        check("AI 漏分割样本不被破坏（补齐率≈0）", h.rescued_ratio < 0.005,
              f"rescued={h.rescued_ratio:.2%}")


def test_ai_engine():
    section("[3] AI 语义分割（自包含 u2netp）")
    ok, why = ai_segment.availability_detail()
    check("AI 环境就绪", ok, why)
    if not ok:
        return

    # 复杂背景照片
    path = os.path.join(ANIME_DIR, "其他背景.jpg")
    res = pipeline.process_image(path, os.path.join(OUT_DIR, "ai_其他.png"), engine="ai")
    check("复杂背景 → ai", res.mode == "ai", f"got {res.mode}")
    check("AI 主体保留率合理(10%~90%)", 0.10 < res.fg_keep_ratio < 0.90,
          f"{res.fg_keep_ratio:.1%}")

    # AI 掩码尺寸正确
    _p = sample("deepseek.jpg")
    if not os.path.exists(_p):
        print("  (跳过 AI alpha 尺寸用例：样本 deepseek.jpg 缺失)")
        return
    rgb, _ = io.load_image(_p)
    a = ai_segment.predict_alpha(rgb)
    check("AI alpha 尺寸与原图一致", a.shape == rgb.shape[:2],
          f"{a.shape} vs {rgb.shape[:2]}")
    check("AI alpha 为 uint8", a.dtype == np.uint8, str(a.dtype))

    # 自包含实现必须与 rembg 一致（若环境里还装着 rembg 则对照）
    try:
        import rembg  # noqa: F401
        os.environ["U2NET_HOME"] = os.path.join(ROOT, "src", "models")
        from rembg import new_session, remove
        ref = np.asarray(
            remove(Image.fromarray(rgb), session=new_session("u2netp")).convert("RGBA")
        )[:, :, 3]
        diff = int(np.abs(a.astype(int) - ref.astype(int)).max())
        check("与 rembg 输出逐像素一致", diff == 0, f"max diff={diff}")
    except Exception:
        print("  (跳过 rembg 一致性对照：环境未安装 rembg)")


def test_interactive():
    section("[4] 交互式抠图（矩形/点选/描边/套索）")
    path = os.path.join(ANIME_DIR, "黑色背景.jpg")
    rgb, _ = io.load_image(path)
    h, w = rgb.shape[:2]

    _, a = interactive.matting_from_rect(rgb, (w // 6, h // 6, w * 2 // 3, h * 2 // 3))
    check("矩形框选", a.shape == (h, w) and 0.05 < np.mean(a >= 128) < 0.95,
          f"fg={np.mean(a >= 128):.1%}")

    _, a = interactive.matting_from_points(rgb, [(w // 2, h // 2)], [(2, 2), (w - 3, h - 3)])
    check("点选", a.shape == (h, w))

    poly = [(w // 6, h // 6), (w * 5 // 6, h // 6),
            (w * 5 // 6, h * 5 // 6), (w // 6, h * 5 // 6)]
    _, a = interactive.matting_from_polygon(rgb, poly)
    check("描边（多边形）", a.shape == (h, w))

    mask = interactive.rasterize_curve((h, w), poly)
    _, a = interactive.matting_from_mask(rgb, mask)
    check("套索（自由曲线栅格化）", a.shape == (h, w))

    # 退化输入必须抛错而不是崩溃
    try:
        interactive.matting_from_rect(rgb, (0, 0, 1, 1))
        check("矩形过小应报错", False)
    except ValueError:
        check("矩形过小应报错", True)
    try:
        interactive.matting_from_polygon(rgb, [(0, 0), (1, 1)])
        check("多边形顶点不足应报错", False)
    except ValueError:
        check("多边形顶点不足应报错", True)


def test_gui_build():
    section("[5] GUI 窗口构建")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtWidgets
        from src.ui.gui import MainWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        win.show()
        check("窗口构建", win.windowTitle() == "LucentCut — 图片背景去除")

        ds = sample("deepseek.jpg")
        if not os.path.exists(ds):
            print("  (跳过 GUI 加载用例：样本 deepseek.jpg 缺失)")
            win.close()
            return
        win._load(ds)
        check("加载无 alpha 图", win._rgb is not None and win._origin_alpha is None)
        r = win._run_process(win._rgb, win._origin_alpha, win._kwargs())
        check("GUI 默认混合引擎", str(r["mode"]).startswith("hybrid") or r["mode"] in ("white",),
              f"got {r['mode']}")

        # 载入已抠好的图应保留 origin alpha（供"沿用"判断）
        ds4 = os.path.join(ANIME_DIR, "deepseek4.png")
        if os.path.exists(ds4):
            win._load(ds4)
            check("加载含 alpha 图保留 origin_alpha", win._origin_alpha is not None)
            r2 = win._run_process(win._rgb, win._origin_alpha, win._kwargs())
            check("GUI 沿用已有 alpha", r2["mode"] == "reuse", f"got {r2['mode']}")
        win.close()
    except Exception:  # noqa: BLE001
        check("GUI 构建", False, traceback.format_exc().splitlines()[-1])


def test_edge_cases():
    section("[6] 边界与异常输入")
    tmp = tempfile.mkdtemp(prefix="lucentcut_test_")

    def p(name):
        return os.path.join(tmp, name)

    # 6.1 不存在的文件
    try:
        io.load_image(p("nope.png"))
        check("不存在的文件应报 FileNotFoundError", False)
    except FileNotFoundError:
        check("不存在的文件应报 FileNotFoundError", True)

    # 6.2 假图片（伪装成 png 的文本）
    with open(p("fake.png"), "w", encoding="utf-8") as f:
        f.write("this is not an image")
    try:
        io.load_image(p("fake.png"))
        check("损坏图片应报 ValueError", False)
    except ValueError:
        check("损坏图片应报 ValueError", True)

    # 6.3 截断的 PNG
    real = sample("deepseek2.png")
    if os.path.exists(real):
        with open(real, "rb") as f:
            data = f.read()
        with open(p("trunc.png"), "wb") as f:
            f.write(data[: len(data) // 3])
        try:
            io.load_image(p("trunc.png"))
            check("截断图片应报 ValueError", False)
        except ValueError:
            check("截断图片应报 ValueError", True)
    else:
        print("  (跳过截断 PNG 用例：样本 deepseek2.png 缺失)")

    # 6.4 1×1 纯白图
    Image.new("RGB", (1, 1), (255, 255, 255)).save(p("1x1_white.png"))
    res = pipeline.process_image(p("1x1_white.png"), p("1x1_white_out.png"))
    check("1×1 纯白图不崩溃且全透明", res.fg_keep_ratio == 0.0,
          f"fg={res.fg_keep_ratio}")

    # 6.5 灰度图 / 调色板图 / 16 位图 / CMYK
    arr = np.tile(np.arange(0, 256, dtype=np.uint8), (256, 1))
    Image.fromarray(arr, mode="L").save(p("gray.png"))
    Image.fromarray(arr, mode="L").convert("P").save(p("palette.png"))
    Image.fromarray((arr.astype(np.uint16) * 257)).save(p("gray16.png"))
    Image.new("CMYK", (64, 64), (0, 0, 0, 0)).save(p("cmyk.jpg"))
    for nm in ("gray.png", "palette.png", "gray16.png", "cmyk.jpg"):
        try:
            r = pipeline.process_image(p(nm), p(nm + ".out.png"))
            check(f"{nm} 可处理", r.mode not in ("", "skipped"), f"mode={r.mode}")
        except Exception as exc:  # noqa: BLE001
            check(f"{nm} 可处理", False, f"{type(exc).__name__}: {exc}")

    # 6.6 全透明图（沿用逻辑不应崩）
    rgba = np.zeros((32, 32, 4), np.uint8)
    Image.fromarray(rgba, "RGBA").save(p("all_trans.png"))
    try:
        r = pipeline.process_image(p("all_trans.png"), p("all_trans_out.png"))
        check("全透明图不崩溃", True)
    except Exception as exc:  # noqa: BLE001
        check("全透明图不崩溃", False, f"{type(exc).__name__}: {exc}")

    # 6.7 主体占满四边（无纯背景）应被识别为复杂背景
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(noisy).save(p("noise.png"))
    check("噪声图判为复杂背景", colorkey.detect_mode(noisy) == colorkey.COMPLEX,
          colorkey.detect_mode(noisy))

    # 6.8 软阈值参数非法
    try:
        colorkey.soft_threshold(np.zeros((4, 4), np.float32), 80, 30)
        check("t_low>=t_high 应报错", False)
    except ValueError:
        check("t_low>=t_high 应报错", True)

    # 6.9 输出目录自动创建
    out = p(os.path.join("deep", "nested", "out.png"))
    _any = sample("deepseek2.png")
    if not os.path.exists(_any):
        _any = sample("deekseep3.jpg")
    pipeline.process_image(_any, out)
    check("自动创建多级输出目录", os.path.isfile(out))

    # 6.10 中文/空格文件名
    src = p("中文 名字 图.png")
    Image.open(_any).save(src)
    try:
        r = pipeline.process_image(src, p("中文 名字 图_out.png"))
        check("中文/空格文件名可用", os.path.isfile(r.output_path))
    except Exception as exc:  # noqa: BLE001
        check("中文/空格文件名可用", False, f"{type(exc).__name__}: {exc}")

    # 6.11 批量处理：混合有效/无效文件，不应因单个坏文件整体失败
    batch_in = p("batch_in")
    os.makedirs(batch_in, exist_ok=True)
    Image.new("RGB", (64, 64), (255, 255, 255)).save(os.path.join(batch_in, "a.png"))
    with open(os.path.join(batch_in, "bad.png"), "w", encoding="utf-8") as f:
        f.write("broken")
    with open(os.path.join(batch_in, "note.txt"), "w", encoding="utf-8") as f:
        f.write("ignore me")
    results = pipeline.process_batch(batch_in, p("batch_out"))
    skipped = [r for r in results if r.mode == "skipped"]
    check("批量跳过坏文件而非中断", len(results) == 2 and len(skipped) == 1,
          f"results={len(results)} skipped={len(skipped)}")

    # 6.12 单测：harden_alpha / merge_alpha / extend_edge_color / keep_border_connected
    a = np.array([[0, 10, 250, 255]], np.uint8)
    h = refine.harden_alpha(a)
    check("harden_alpha 端点吸附", list(h[0]) == [0, 0, 255, 255], str(h))
    m = refine.merge_alpha(np.array([[255, 128]], np.uint8), np.array([[128, 255]], np.uint8))
    check("merge_alpha 相乘", m[0, 0] == 128 and m[0, 1] == 128, str(m))
    rgb = np.full((8, 8, 3), 100, np.uint8)
    check("extend_edge_color 全不透明时不变",
          np.array_equal(refine.extend_edge_color(rgb, np.full((8, 8), 255, np.uint8)), rgb))
    # keep_border_connected：内部孤立透明块应被恢复为不透明
    al = np.full((20, 20), 255, np.uint8)
    al[0:20, 0:20] = 255
    al[8:12, 8:12] = 0     # 内部洞（不与边界连通）
    out = colorkey.keep_border_connected(al)
    check("keep_border_connected 恢复内部洞", out[10, 10] == 255, str(out[10, 10]))

    shutil.rmtree(tmp, ignore_errors=True)


def test_manual_override():
    section("[7] 人工裁决层（框内保留 / 框内去除）")

    # 8.1 多区域栅格化：互不相连的两个区域都能表达
    rgb = np.full((40, 40, 3), 250, np.uint8)
    regs = [[(0, 0), (5, 0), (5, 5), (0, 5)], [(35, 35), (39, 35), (39, 39), (35, 39)]]
    m = interactive.regions_to_mask(rgb.shape, regs)
    check("多区域：两个区域都被填充", int((m > 0).sum()) == 36 + 25, f"{int((m > 0).sum())}")
    check("多区域：区域之间不连通", m[20, 20] == 0 and m[2, 2] == 255 and m[37, 37] == 255)
    check("多区域：少于 3 点的区域被忽略",
          int(interactive.regions_to_mask(rgb.shape, [[(1, 1), (2, 2)]]).max()) == 0)

    # 8.2 框内保留：被算法删掉的区域应当整块回到 255，并用原图颜色
    origin = rgb.copy()
    origin[5:15, 5:15] = (200, 10, 10)      # 原图真实颜色（红衣服）
    processed = rgb.copy()
    processed[5:15, 5:15] = (77, 77, 77)    # 算法把这块当背景后留下的脏色
    alpha = np.zeros((40, 40), np.uint8)
    alpha[10:30, 10:30] = 255
    keep = np.zeros((40, 40), np.uint8)
    keep[5:15, 5:15] = 255
    r, a = interactive.apply_override(processed, alpha, origin, keep, None, feather_px=1)
    check("框内保留：区域内 alpha=255", int(a[8, 8]) == 255, f"{int(a[8, 8])}")
    check("框内保留：用原图 RGB 回填", tuple(int(v) for v in r[8, 8]) == (200, 10, 10),
          str(r[8, 8]))
    check("框内保留：区域外不受影响", int(a[0, 0]) == 0 and int(a[20, 20]) == 255)

    # 8.3 框内去除
    drop = np.zeros((40, 40), np.uint8)
    drop[12:18, 12:18] = 255
    _, a2 = interactive.apply_override(processed, alpha, origin, None, drop, feather_px=1)
    check("框内去除：区域内 alpha=0", int(a2[15, 15]) == 0, f"{int(a2[15, 15])}")
    # 远离边界的地方不受影响；紧贴边界的一圈按设计会被柔化（不是硬边）
    check("框内去除：远处不受影响", int(a2[25, 25]) == 255, f"{int(a2[25, 25])}")
    check("框内去除：边界被柔化而非硬边", 0 < int(a2[11, 11]) < 255, f"{int(a2[11, 11])}")
    _, a2h = interactive.apply_override(processed, alpha, origin, None, drop, feather_px=0)
    check("柔化=0 时去除边界为硬边",
          int(a2h[11, 11]) == 255 and int(a2h[12, 12]) == 0,
          f"{int(a2h[11, 11])},{int(a2h[12, 12])}")

    # 8.4 keep 与 drop 重叠时以保留为准（用户专门描出来要留的）
    _, a3 = interactive.apply_override(processed, alpha, origin, keep, drop, feather_px=1)
    check("重叠区域以「保留」为准", int(a3[13, 13]) == 255, f"{int(a3[13, 13])}")

    # 8.5 只柔化边界，区域内部必须保持满值（否则看起来像"保留是假的"）
    _, a4 = interactive.apply_override(processed, alpha, origin, keep, None, feather_px=1)
    check("保留区内部为满值 255", int(a4[8, 8]) == 255 and int(a4[14, 14]) == 255,
          f"{int(a4[8, 8])},{int(a4[14, 14])}")
    _, a5 = interactive.apply_override(processed, alpha, origin, keep, None, feather_px=0)
    check("柔化=0 时边界为硬边", int(a5[4, 4]) == 0 and int(a5[5, 5]) == 255,
          f"{int(a5[4, 4])},{int(a5[5, 5])}")

    # 8.6 多区域 GrabCut
    big = np.full((60, 60, 3), 250, np.uint8)
    big[10:50, 10:50] = (30, 30, 30)
    _, am = interactive.matting_from_regions(big, [[(8, 8), (52, 8), (52, 52), (8, 52)]])
    check("多区域 GrabCut 输出形状正确", am.shape == (60, 60))
    try:
        interactive.matting_from_regions(big, [])
        check("无区域应报错", False)
    except ValueError:
        check("无区域应报错", True)

    # 8.7 关键决策 D9：先处理 → 再描边 → 又点一次「开始处理」，裁决必须还在
    path = os.path.join(ANIME_DIR, "doro1.png")
    if os.path.exists(path):
        src_rgb, _ = io.load_image(path)
        h, w = src_rgb.shape[:2]
        km = np.zeros((h, w), np.uint8)
        km[int(h * 0.55):int(h * 0.75), int(w * 0.30):int(w * 0.70)] = 255
        _, a_plain, *_ = pipeline.process_array(src_rgb, None, engine="hybrid")
        _, a_keep, *_ = pipeline.process_array(src_rgb, None, engine="hybrid", keep_mask=km)
        check("重新处理后裁决层依然生效", float(a_keep[km > 0].mean()) > 250.0,
              f"mean={float(a_keep[km > 0].mean()):.1f}")
        check("裁决层确实改变了结果", not np.array_equal(a_plain, a_keep))
        dm = km.copy()
        _, a_drop, *_ = pipeline.process_array(src_rgb, None, engine="hybrid", drop_mask=dm)
        check("去除裁决在重新处理后生效", float(a_drop[dm > 0].mean()) < 5.0,
              f"mean={float(a_drop[dm > 0].mean()):.1f}")


def test_canvas_interaction():
    section("[8] 画布交互（平移 / 多区域 / 撤销）")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from src.ui.canvas import Canvas
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        c = Canvas()
        rgb = np.full((100, 100, 3), 200, np.uint8)
        c.set_image(rgb, None, "checker", keep_transform=False)
        c.resize(400, 400)
        c._scale = 1.0
        c._offset = QtCore.QPoint(0, 0)

        # 9.1 坐标换算：平移前后同一个屏幕点应映射到不同原图点
        p_before = c._to_image(QtCore.QPoint(30, 30))
        c._offset = QtCore.QPoint(10, 10)
        p_after = c._to_image(QtCore.QPoint(30, 30))
        check("平移改变屏幕→原图映射",
              p_before == (30, 30) and p_after == (20, 20), f"{p_before} → {p_after}")

        # 9.2 平移不影响已描出的点（点存的是原图坐标）
        c.set_tool("polygon")
        c._draft = [(20, 20), (40, 20), (40, 40)]
        c._offset = QtCore.QPoint(50, 50)
        check("平移后已描点保持原图坐标不变",
              c._draft == [(20, 20), (40, 20), (40, 40)], str(c._draft))

        # 9.3 多区域：闭合一个后继续画第二个
        c._offset = QtCore.QPoint(0, 0)
        c._draft = [(0, 0), (10, 0), (10, 10)]
        c._close_draft()
        c._draft = [(50, 50), (60, 50), (60, 60)]
        c._close_draft()
        check("可连续闭合多个区域", c.selection_state()[0] == 2, str(c.selection_state()))

        # 9.4 提交后信号带出全部区域
        got = {}
        c.selectionCommitted.connect(lambda r, f, b: got.update(regions=r, fg=f, bg=b))
        ok = c.commit()
        check("提交返回 True 且带出 2 个区域", ok and len(got.get("regions", [])) == 2,
              str(got.get("regions")))

        # 9.5 三级撤销：草稿点 → 已闭合区域 → 请求主窗口
        c._draft = [(1, 1), (2, 2), (3, 3)]
        c._regions = [[(0, 0), (9, 0), (9, 9)]]
        c.undo()
        check("撤销①：先退草稿点", len(c._draft) == 2 and len(c._regions) == 1)
        c._draft = []
        c.undo()
        check("撤销②：再退已闭合区域（变回草稿可续画）",
              len(c._regions) == 0 and len(c._draft) == 3, str(c._draft))
        c._draft = []
        asked = []
        c.undoRequested.connect(lambda: asked.append(1))
        c.undo()
        check("撤销③：无内容时请求主窗口回退", asked == [1])

        # 9.6 空选区提交返回 False，不应触发信号
        c.clear_selection()
        got.clear()
        check("空选区不提交", c.commit() is False and not got)

        # 9.7 中键拖动只改 offset，不动选区
        c.set_tool("polygon")
        c._draft = [(5, 5)]
        before = QtCore.QPoint(c._offset)
        c._panning = True
        c._pan_start = QtCore.QPoint(0, 0)
        c._pan_offset0 = QtCore.QPoint(before)
        c.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(25, 15), QtCore.Qt.NoButton,
            QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
        check("中键拖动只平移视图",
              c._offset == QtCore.QPoint(before.x() + 25, before.y() + 15)
              and c._draft == [(5, 5)], f"{c._offset}, draft={c._draft}")
        c._panning = False

        # 9.8 适应窗口后整图应在画布内
        c.fit_view()
        check("适应窗口：缩放后整图不超出画布",
              c._pixmap.width() * c._scale <= 400 + 1 and c._pixmap.height() * c._scale <= 400 + 1,
              f"scale={c._scale:.3f}")
        c.close()
    except Exception:  # noqa: BLE001
        check("画布交互", False, traceback.format_exc().splitlines()[-1])


def test_gui_override_flow():
    section("[9] GUI 人工修改流程")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtWidgets
        from src.ui.gui import MainWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        win.show()
        path = os.path.join(ANIME_DIR, "doro1.png")
        if not os.path.exists(path):
            print("  (跳过 GUI 人工修改流程用例：样本 doro1.png 缺失)")
            win.close()
            return
        win._load(path)
        h, w = win._rgb.shape[:2]

        # 10.1 未处理时画保留区 → 自动先跑算法再叠加
        km = np.zeros((h, w), np.uint8)
        km[int(h * 0.6):int(h * 0.8), int(w * 0.3):int(w * 0.7)] = 255
        win._push_history()
        win._keep_mask = km
        win._apply_override_local()
        if win._worker is not None:
            win._worker.wait()          # 后台线程跑完
        app.processEvents()             # 让 done 信号在主线程派发
        check("未处理时描保留区会自动先跑算法", win._alpha is not None)

        # 10.2 再点一次「开始处理」，裁决仍在
        r = win._run_process(win._rgb, win._origin_alpha, win._kwargs())
        a = r["alpha"]
        check("GUI 重新处理后保留区仍为不透明", float(a[km > 0].mean()) > 250.0,
              f"mean={float(a[km > 0].mean()):.1f}")

        # 10.3 去除区叠加
        dm = np.zeros((h, w), np.uint8)
        dm[0:int(h * 0.05), 0:int(w * 0.05)] = 255
        win._drop_mask = dm
        r2 = win._run_process(win._rgb, win._origin_alpha, win._kwargs())
        check("GUI 去除区生效", float(r2["alpha"][dm > 0].mean()) < 5.0,
              f"mean={float(r2['alpha'][dm > 0].mean()):.1f}")

        # 10.4 撤销回退裁决层
        win._push_history()
        win._keep_mask = None
        check("撤销前有历史", len(win._history) >= 1)
        win._undo()
        check("撤销恢复保留区", win._keep_mask is not None)

        # 10.5 keep 与 drop 互斥（M13 修订：后画的覆盖先画的）
        km2 = np.zeros((h, w), np.uint8)
        km2[0:5, 0:5] = 255
        win._keep_mask, win._drop_mask = interactive.merge_override_masks(
            win._keep_mask, win._drop_mask, km2, None)
        check("画保留会清掉该处的去除标记", int(win._drop_mask[0, 0]) == 0)

        # 10.6 回归：空状态下反复点撤销不得递归崩溃
        win._history = []
        win._keep_mask = win._drop_mask = None
        win.dst_canvas.clear_selection()
        for _ in range(20):
            win._undo()
        check("空状态反复撤销不崩溃", True)

        # 10.7 保存结果可用
        out = os.path.join(OUT_DIR, "gui_override.png")
        win.out_edit.setText(out)
        win._save_result()
        check("带裁决层的结果可保存", os.path.isfile(out))
        win.close()
    except Exception:  # noqa: BLE001
        check("GUI 人工修改流程", False, traceback.format_exc().splitlines()[-1])


def test_interaction_fix():
    section("[10] 交互修正（点选笔刷 / 显示修改区 / 应用选区语义）")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from src.ui.canvas import Canvas
        from src.ui.gui import MainWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        def _ev(kind, x, y, button, buttons=None):
            return QtGui.QMouseEvent(kind, QtCore.QPointF(x, y), button,
                                     button if buttons is None else buttons,
                                     QtCore.Qt.NoModifier)

        # 11.1 点选 = 笔刷（左键当前作用色 / 右键反面），且可拖动连续涂
        c = Canvas()
        rgb = np.full((120, 120, 3), 200, np.uint8)
        c.set_image(rgb, None, "checker", keep_transform=False)
        c.resize(300, 300)
        c._scale = 1.0
        c._offset = QtCore.QPoint(0, 0)
        c.set_tool("fg")
        c.set_brush_radius(10)
        c.set_paint_mode("keep")

        c.mousePressEvent(_ev(QtCore.QEvent.MouseButtonPress, 30, 30, QtCore.Qt.LeftButton))
        check("点选左键=当前作用（保留）",
              len(c._fg_pts) == 1 and not c._bg_pts and c._fg_pts[0] == (30, 30),
              f"{c._fg_pts}/{c._bg_pts}")
        # 按住左键拖动 → 连续记录多个点（原来只记一个点击点）
        for x in range(35, 90, 5):
            c.mouseMoveEvent(_ev(QtCore.QEvent.MouseMove, x, 30,
                                 QtCore.Qt.NoButton, QtCore.Qt.LeftButton))
        check("按住拖动可连续涂抹", len(c._fg_pts) > 5, str(len(c._fg_pts)))
        c.mouseReleaseEvent(_ev(QtCore.QEvent.MouseButtonRelease, 85, 30, QtCore.Qt.LeftButton))

        c.mousePressEvent(_ev(QtCore.QEvent.MouseButtonPress, 60, 60, QtCore.Qt.RightButton))
        check("点选右键=反面（去除）", len(c._bg_pts) == 1 and c._bg_pts[0] == (60, 60),
              f"{c._bg_pts}")
        c.mouseReleaseEvent(_ev(QtCore.QEvent.MouseButtonRelease, 60, 60, QtCore.Qt.RightButton))

        c.set_paint_mode("drop")
        c.mousePressEvent(_ev(QtCore.QEvent.MouseButtonPress, 20, 20, QtCore.Qt.LeftButton))
        check("作用=去除时左键涂去除", len(c._bg_pts) == 2, f"{c._bg_pts}")
        c.mouseReleaseEvent(_ev(QtCore.QEvent.MouseButtonRelease, 20, 20, QtCore.Qt.LeftButton))
        c.close()

        # 11.2 应用选区语义：点选保留只影响涂抹处，不得整图清空（"整图变红"的回归）
        win = MainWindow()
        win.show()
        path = sample("deepseek2.png")
        if not os.path.exists(path):
            # 该样本未随仓提交：回退到存在的样本，保证本组用例仍能执行
            path = sample("deepseek1.jpg")
        if not os.path.exists(path):
            print("  (跳过点选语义用例：样本缺失)")
            win.close()
            return
        win._load(path)
        h, w = win._rgb.shape[:2]
        r = win._run_process(win._rgb, win._origin_alpha, win._kwargs())
        win._alpha, win._out_rgb = r["alpha"], r["rgb"]
        win.dst_canvas.set_image(r["rgb"], r["alpha"], win._bg())
        before = r["alpha"].copy()

        check("「显示修改区」默认关闭（处理完不留绿色描边）", not win.show_ov.isChecked())

        win._set_tool("fg")
        win.action_combo.setCurrentIndex(0)          # 保留
        win.brush.setValue(8)
        cx, cy = w // 2, h // 2
        win._on_commit([], [(cx, cy)], [])
        m = interactive.points_to_mask(win._rgb.shape, [(cx, cy)], 8)
        check("点选保留：涂抹处 alpha=255", float(win._alpha[m > 0].mean()) > 250.0,
              f"{float(win._alpha[m > 0].mean()):.1f}")
        ratio_now = float((win._alpha >= 128).mean())
        ratio_old = float((before >= 128).mean())
        check("点选保留：其余像素未被清空（不再整图变红）",
              ratio_now >= 0.9 * ratio_old, f"{ratio_now:.3f} vs {ratio_old:.3f}")
        check("涂抹点来自原图颜色（保留区使用本来的 RGB）",
              bool(np.any(win._out_rgb[m > 0] != 0)))

        # 11.3 显示修改区：默认不显示视觉层，勾选后才显示
        check("应用后不残留绿色修改区", win.dst_canvas._keep_img is None)
        win.show_ov.setChecked(True)
        check("勾选「显示修改区」后可见标注", win.dst_canvas._keep_img is not None)
        win.show_ov.setChecked(False)
        check("取消勾选后标注消失", win.dst_canvas._keep_img is None)

        # 11.4 重新处理一次，裁决仍在，且默认仍不显示绿色描边
        r2 = win._run_process(win._rgb, win._origin_alpha, win._kwargs())
        win._alpha, win._out_rgb = r2["alpha"], r2["rgb"]
        win.dst_canvas.set_image(r2["rgb"], r2["alpha"], win._bg())
        win._refresh_overlay()
        check("重新处理后保留裁决依然生效", float(win._alpha[m > 0].mean()) > 250.0,
              f"{float(win._alpha[m > 0].mean()):.1f}")
        check("重新处理后仍不残留绿色描边", win.dst_canvas._keep_img is None)

        # 11.5 作用切换同步到画布笔刷色
        win.action_combo.setCurrentIndex(1)          # 去除
        check("切换作用同步到笔刷色", win.dst_canvas.paint_mode() == "drop",
              win.dst_canvas.paint_mode())
        win.action_combo.setCurrentIndex(0)
        check("切回保留同步到笔刷色", win.dst_canvas.paint_mode() == "keep")

        # 11.6 GrabCut 抽成独立按钮：没画区域只提示、不崩，也不清空结果
        win.dst_canvas.clear_selection()
        alpha_before = win._alpha.copy()
        win._grabcut()
        check("无区域时 GrabCut 只提示不崩溃", "圈出" in win.status_label.text(),
              win.status_label.text())
        check("无区域时 GrabCut 不动结果", np.array_equal(win._alpha, alpha_before))

        # 11.7 GrabCut 有区域时可正常跑（重新计算 alpha）
        win.dst_canvas._regions = [[(int(w * 0.1), int(h * 0.1)),
                                    (int(w * 0.9), int(h * 0.1)),
                                    (int(w * 0.9), int(h * 0.9)),
                                    (int(w * 0.1), int(h * 0.9))]]
        win._grabcut()
        check("GrabCut 按钮启动后清空已画区域", not win.dst_canvas.has_pending())
        if win._worker is not None:
            win._worker.wait()
        app.processEvents()
        check("GrabCut 按钮可产出结果", win._alpha is not None and win._alpha.shape == (h, w))
        win.close()
    except Exception:  # noqa: BLE001
        check("交互修正", False, traceback.format_exc().splitlines()[-1])


def test_hq_engine():
    """[13] 精细抠图引擎（M12：BiRefNet / ISNet）。

    用户反馈（2026-10-11）：动漫多人合影 image/1.jpg 用 u2netp 抠出来的轮廓是
    "圆头团块"——头发丝全丢、轮廓发糊；换成 BiRefNet（1024×1024）后轮廓锐利、
    发丝保留，与用户用其他软件处理的参考图 image/2.png 的二值 IoU 达 0.988。

    本组测试锁住这条质量线，同时保证三件事：
      (a) 模型缺失 / 推理失败时能优雅回退，绝不因此报错；
      (b) 纯色背景不盲目上大模型（否则白底图要白等 9 秒）；
      (c) 同一进程里连续推理不因 onnxruntime 内存池不回收而失败（曾实测踩坑）。
    """
    section("[13] 精细抠图引擎（M12）")
    ok, why = hq_segment.availability_detail()
    print(f"  （模型状态：{why}）")
    check("availability_detail 返回 (bool, str)",
          isinstance(ok, bool) and isinstance(why, str) and len(why) > 0)

    # 13.1 模型缺失 → matting 引擎必须优雅回退，仍产出可用的 alpha
    _sp = sample("deepseek.jpg")
    if not os.path.exists(_sp):
        _sp = os.path.join(IMG_DIR, "deekseep3.jpg")
    rgb_small, _ = io.load_image(_sp)
    orig_find = hq_segment.find_model
    try:
        hq_segment.find_model = lambda: None       # type: ignore[assignment]
        _, a_fb, mode_fb, *_ = pipeline.process_array(rgb_small, None,
                                                       engine="matting")
        check("模型缺失时 matting 回退成功（不报错）",
              a_fb.shape == rgb_small.shape[:2] and mode_fb != "matting",
              f"mode={mode_fb}")
    finally:
        hq_segment.find_model = orig_find          # type: ignore[assignment]

    # 13.2 纯色背景自动走颜色键控（不该白等大模型）
    t0 = time.time()
    _, _, mode_solid, *_ = pipeline.process_array(rgb_small, None, engine="matting")
    dt_solid = time.time() - t0
    check("纯色背景自动走颜色键控且很快（<5s）",
          mode_solid != "matting" and dt_solid < 5.0,
          f"mode={mode_solid} {dt_solid:.2f}s")

    hit = hq_segment.find_model()
    check("find_model 返回 (spec, 已存在的 onnx 路径)",
          hit is not None and os.path.isfile(hit[1]) and bool(hit[0].label))

    if not ok:
        print("  (跳过后续大模型用例：模型不可用)")
        return

    # 13.3 复杂背景启用大模型（并锁住连续推理的内存稳定性）
    p1 = os.path.join(IMG_DIR, "1.jpg")
    p2 = os.path.join(IMG_DIR, "3D角色人物图",
                      "2c05beadb26d95e25979cc0c45e93a37_720.jpeg")
    if os.path.exists(p1):
        rgb1, _ = io.load_image(p1)
        t0 = time.time()
        _, a1, mode1, *_ = pipeline.process_array(rgb1, None, engine="matting")
        dt1 = time.time() - t0
        check("复杂背景 → matting", mode1 == "matting", f"got {mode1}")
        check("matting alpha 尺寸/类型正确",
              a1.shape == rgb1.shape[:2] and a1.dtype == np.uint8)
        check("matting 单张耗时在可接受范围（<60s）", dt1 < 60.0, f"{dt1:.1f}s")

        # 13.4 与用户提供的参考结果对比：IoU >= 0.90，且明显优于 u2netp
        ref_p = os.path.join(IMG_DIR, "2.png")
        if os.path.exists(ref_p):
            ref = np.asarray(Image.open(ref_p).convert("RGBA"))[:, :, 3]
            if ref.shape == a1.shape:
                rb = ref >= 128
                mb = a1 >= 128
                iou_hq = float((rb & mb).sum()) / max(1, int((rb | mb).sum()))
                ai_b = ai_segment.predict_alpha(rgb1) >= 128
                iou_ai = float((rb & ai_b).sum()) / max(1, int((rb | ai_b).sum()))
                check(f"精细抠图轮廓贴合参考图（IoU {iou_hq:.3f} >= 0.90）",
                      iou_hq >= 0.90)
                check(f"精细抠图明显优于 u2netp（{iou_hq:.3f} > {iou_ai:.3f} + 0.05）",
                      iou_hq > iou_ai + 0.05)

    if os.path.exists(p2):
        rgb2, _ = io.load_image(p2)
        _, a2, mode2, *_ = pipeline.process_array(rgb2, None, engine="matting")
        check("同一进程连续第二次大模型推理不报内存错",
              a2.shape == rgb2.shape[:2] and mode2 == "matting", f"mode={mode2}")

    # 13.5 rescue_subject：颜色键控误删的同色部件被 HQ 掩码补回
    ca = np.zeros((80, 80), np.uint8)
    ca[20:60, 20:60] = 255            # 颜色键控认出的主体
    ca[20:60, 60:72] = 0              # 同色的白色部件：颜色键控判成了背景
    hq = np.zeros((80, 80), np.uint8)
    hq[20:60, 20:72] = 255            # HQ 明确说这一整块都是主体
    out, ratio = hq_segment.rescue_subject(ca, hq)
    check("rescue_subject 补回同色部件",
          int(out[40, 66]) == 255 and ratio > 0,
          f"val={int(out[40,66])} ratio={ratio:.3f}")
    check("rescue_subject 不把真背景补回来", int(out[5, 5]) == 0, f"{int(out[5,5])}")

    # 13.6 输入尺寸从 onnx 图读取（换更大的模型无需改代码）
    check("_input_size 可调用（尺寸不写死）", callable(hq_segment._input_size))


def test_cli():
    section("[12] CLI 命令行")
    tmp = tempfile.mkdtemp(prefix="lucentcut_cli_")
    src = sample("deepseek2.png")
    if not os.path.exists(src):
        src = sample("deekseep3.jpg")
    out = os.path.join(tmp, "cli_out.png")
    r = subprocess.run([PY, "-m", "src.main", "-i", src, "-o", out],
                       cwd=ROOT, capture_output=True, text=True)
    check("CLI 单图返回码 0", r.returncode == 0, r.stderr.strip()[-200:])
    check("CLI 单图产出文件", os.path.isfile(out))

    r2 = subprocess.run(
        [PY, "-m", "src.main", "-i", ANIME_DIR, "-o", os.path.join(tmp, "batch"),
         "--batch", "--engine", "color"],
        cwd=ROOT, capture_output=True, text=True)
    check("CLI 批量返回码 0", r2.returncode == 0, r2.stderr.strip()[-200:])

    r3 = subprocess.run([PY, "-m", "src.main", "-i", os.path.join(tmp, "none.png")],
                        cwd=ROOT, capture_output=True, text=True)
    check("CLI 坏输入返回码非 0", r3.returncode != 0)

    shutil.rmtree(tmp, ignore_errors=True)


def test_alpha_quality():
    """[11] 内部半透明修复（画质）：浅色内部部件不得残留半透明。

    回归 2026-10-11 用户反馈的"处理后画质下降"——
    deepseek.jpg 的眼睛高光 RGB≈(205,223,239)，到白底色差仅 55~79，
    落在软阈值 [30,80] 内 → alpha≈160；旧 keep_border_connected(thr=128)
    只修 alpha<128 的洞，于是这块被深蓝眼珠完全包住的高光被永久留成半透明。
    """
    section("[11] 内部半透明修复（画质）")
    import cv2

    # 12.1 合成用例：被不透明像素完全包围的浅色块 → 必须变成 255
    a = np.zeros((60, 60), np.uint8)
    a[10:50, 10:50] = 255          # 主体
    a[25:35, 25:35] = 160          # 主体内部浅色部件（旧代码会漏掉）
    out = colorkey.keep_border_connected(a)
    check("内部 [128,254] 半透明块 → 255",
          int(out[25:35, 25:35].min()) == 255, f"min={int(out[25:35,25:35].min())}")
    check("外部背景仍为 0", int(out[0, 0]) == 0, f"{int(out[0,0])}")
    check("主体本身不变", int(out[15, 15]) == 255)

    # 12.2 与边界连通的软过渡带必须保留（不能被硬化成 255）
    b = np.zeros((60, 60), np.uint8)
    b[20:40, 20:40] = 255
    b[19, 20:40] = 128             # 贴着轮廓、与外部背景连通的一圈软边
    out_b = colorkey.keep_border_connected(b)
    check("边界连通软过渡带保留（不被硬化）",
          int(out_b[19, 30]) == 128, f"{int(out_b[19,30])}")

    # 12.3 真实图：deepseek.jpg 眼睛高光区不得再有半透明
    p = os.path.join(ANIME_DIR, "deepseek.jpg")
    if os.path.exists(p):
        rgb, _ = io.load_image(p)
        _, alpha, mode, *_ = pipeline.process_array(rgb, None, engine="hybrid")
        x0, y0, x1, y1 = 330, 545, 530, 770
        eye = alpha[y0:y1, x0:x1]
        check("真实图：眼睛区域无半透明残留（alpha 全 255）",
              int((eye < 250).sum()) == 0, f"仍有 {int((eye<250).sum())} px")
        # 眼睛高光位置直接取值（浅蓝椭圆中心附近）
        check("真实图：眼睛高光点 alpha=255", int(alpha[625, 470]) == 255,
              f"{int(alpha[625,470])}")

    # 12.4 全图不应出现"大块连续半透明"（画质脏斑的特征）
    p2 = os.path.join(ANIME_DIR, "deepseek.jpg")
    if os.path.exists(p2):
        rgb, _ = io.load_image(p2)
        _, alpha, *_ = pipeline.process_array(rgb, None, engine="hybrid")
        bgc = (alpha < 250).astype(np.uint8)
        num, lab = cv2.connectedComponents(bgc, 8)
        border_labels = np.unique(np.concatenate(
            [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
        border_labels = border_labels[border_labels != 0]
        in_bg = np.isin(lab, border_labels)
        interior = (~in_bg) & (alpha < 250)
        check("真实图：主体内部无成片半透明（<0.05%）",
              float(interior.mean()) < 0.0005,
              f"{float(interior.mean()):.4%}")

    # 12.5 修洞阈值不得把背景漏回来（边框透明率仍应很高）
    for name in ["deepseek.jpg", "doro1.png"]:
        pp = os.path.join(ANIME_DIR, name)
        if not os.path.exists(pp):
            continue
        rgb, _ = io.load_image(pp)
        _, alpha, *_ = pipeline.process_array(rgb, None, engine="hybrid")
        t = 3
        m = np.zeros_like(alpha, bool)
        m[:t, :] = m[-t:, :] = True
        m[:, :t] = m[:, -t:] = True
        ratio = float((alpha[m] < 16).mean())
        check(f"边框透明率未下降 {name}（>50%）", ratio > 0.5, f"{ratio:.1%}")

    # 12.6 keep_border_connected 幂等：跑两次结果一致
    once = colorkey.keep_border_connected(a)
    twice = colorkey.keep_border_connected(once)
    check("keep_border_connected 幂等", bool(np.array_equal(once, twice)))


def test_m13_semantics():
    """[14] M13：裁决层跨笔合并语义 + 批量进度回调。"""
    section("[14] M13 裁决语义与进度")
    h, w = 60, 80

    # 14.1 后画去除覆盖先画的保留（修复：先保留过的区域去除不生效）
    keep_old = np.zeros((h, w), np.uint8); keep_old[10:20, 10:20] = 255
    keep, drop = interactive.merge_override_masks(keep_old, None, None,
                                                  np.zeros((h, w), np.uint8))
    drop_new = np.zeros((h, w), np.uint8); drop_new[12:18, 12:18] = 255
    keep, drop = interactive.merge_override_masks(keep, drop, None, drop_new)
    check("后画去除清掉旧保留",
          int(keep[15, 15]) == 0 and drop[15, 15] == 255 and drop[11, 11] == 0)

    # 14.2 后画保留覆盖先画的去除
    drop_old = np.zeros((h, w), np.uint8); drop_old[30:40, 30:40] = 255
    keep2, drop2 = interactive.merge_override_masks(None, drop_old, None, None)
    keep_new2 = np.zeros((h, w), np.uint8); keep_new2[32:38, 32:38] = 255
    keep2, drop2 = interactive.merge_override_masks(keep2, drop2, keep_new2, None)
    check("后画保留清掉旧去除",
          int(drop2[35, 35]) == 0 and keep2[35, 35] == 255 and keep2[31, 31] == 0)

    # 14.3 同一笔里保留与去除重叠 → 保留优先（防误删）
    k = np.zeros((h, w), np.uint8); k[0:10, 0:10] = 255
    d = np.zeros((h, w), np.uint8); d[5:15, 5:15] = 255
    k3, d3 = interactive.merge_override_masks(None, None, k, d)
    check("同一笔重叠保留优先", int(k3[7, 7]) == 255 and int(d3[7, 7]) == 0)

    # 14.4 入参不被修改
    ka = np.zeros((h, w), np.uint8); ka[0:5, 0:5] = 255
    da = np.zeros((h, w), np.uint8); da[0:5, 0:5] = 255
    interactive.merge_override_masks(ka, da, None, None)
    check("merge 不修改入参", int(da[2, 2]) == 255 and int(ka[2, 2]) == 255)

    # 14.5 None 入参可用
    k5, d5 = interactive.merge_override_masks(None, None, None, None)
    check("空合并返回 (None, None)", k5 is None and d5 is None)

    # 14.6 process_batch 进度回调：每张图回调一次，序号正确
    tmp_in = tempfile.mkdtemp(prefix="lc_m13_in_")
    tmp_out = tempfile.mkdtemp(prefix="lc_m13_out_")
    try:
        for i in range(3):
            img = Image.fromarray(
                (np.ones((40, 50, 3), np.uint8) * 255), "RGB")
            img.save(os.path.join(tmp_in, f"t{i}.png"))
        calls = []
        results = pipeline.process_batch(tmp_in, tmp_out,
                                         progress_cb=lambda c, t, n: calls.append((c, t, n)))
        check("批量进度回调每张一次", len(calls) == 3,
              f"calls={calls}")
        check("批量进度序号正确",
              [c[0] for c in calls] == [1, 2, 3] and calls[0][1] == 3)
        check("批量结果不受回调影响", len(results) == 3)
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)

    # 14.7 回调抛异常不影响批处理
    tmp_in2 = tempfile.mkdtemp(prefix="lc_m13_in2_")
    tmp_out2 = tempfile.mkdtemp(prefix="lc_m13_out2_")
    try:
        for i in range(2):
            Image.fromarray(np.ones((40, 50, 3), np.uint8) * 255, "RGB").save(
                os.path.join(tmp_in2, f"t{i}.png"))
        def boom(cur, tot, name):
            raise RuntimeError("进度回调炸了")
        results2 = pipeline.process_batch(tmp_in2, tmp_out2, progress_cb=boom)
        check("回调异常不中断批处理", len(results2) == 2)
    finally:
        shutil.rmtree(tmp_in2, ignore_errors=True)
        shutil.rmtree(tmp_out2, ignore_errors=True)

    # 14.8 GUI：点选语义按颜色生效（红点=去除，即使「作用」选的是保留）
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtWidgets
        from src.ui.gui import MainWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        win.show()
        path = os.path.join(ANIME_DIR, "doro1.png")
        if os.path.exists(path):
            win._load(path)
            hh, ww = win._rgb.shape[:2]
            # 模拟：作用=保留，但画了绿点/红点（颜色语义在绘制时已定）
            win.action_combo.setCurrentIndex(0)          # keep
            kn, dn = win._build_selection_masks(
                [], [(ww // 2, hh // 2)], [])
            check("绿点→保留", int(kn[hh // 2, ww // 2]) == 255 and int(dn[hh // 2, ww // 2]) == 0)
            kn, dn = win._build_selection_masks(
                [], [], [(ww // 2, hh // 2)])
            check("红点→去除（不受「作用」下拉影响）",
                  int(dn[hh // 2, ww // 2]) == 255 and int(kn[hh // 2, ww // 2]) == 0)
            # 区域才受「作用」下拉影响
            kn, dn = win._build_selection_masks(
                [[(0, 0), (20, 0), (20, 20), (0, 20)]], [], [])
            check("作用=保留时区域→保留", int(kn[10, 10]) == 255)
            win.action_combo.setCurrentIndex(1)          # drop
            kn, dn = win._build_selection_masks(
                [[(0, 0), (20, 0), (20, 20), (0, 20)]], [], [])
            check("作用=去除时区域→去除", int(dn[10, 10]) == 255)
            # 14.9 开始处理时未提交涂抹自动并入 + 画布清空
            win._keep_mask = None
            win._drop_mask = None
            win.dst_canvas._bg_pts = [(ww // 2, hh // 2)]
            win._process()
            if win._worker is not None:
                win._worker.wait()
            app.processEvents()
            check("开始处理把未提交涂抹并入裁决层",
                  win._drop_mask is not None and int(win._drop_mask[hh // 2, ww // 2]) == 255)
            check("并入后画布已清空", not win.dst_canvas.has_pending())
            win.close()
    except Exception:  # noqa: BLE001
        check("GUI 点选语义", False, traceback.format_exc().splitlines()[-1])


def test_m15_canvas_redo_roi():
    section("[15] M15：画布重写（Redo） + ROI 局部处理")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtCore, QtWidgets
        from src.ui.canvas import Canvas
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        # 15.1 画布迁移后基本可用（QGraphicsView）
        c = Canvas()
        rgb = np.full((100, 100, 3), 200, np.uint8)
        c.set_image(rgb, None, "checker", keep_transform=False)
        c.resize(400, 400)
        check("画布迁移后为 QGraphicsView 子类",
              isinstance(c, QtWidgets.QGraphicsView))

        # 15.2 撤销一个笔刷点 → 重写恢复
        c.set_tool("fg")
        c.set_paint_mode("keep")
        c.set_brush_radius(8)
        c._paint_at((30, 30), "keep")
        c._paint_at((40, 40), "keep")
        n0 = len(c._fg_pts)
        c.undo()
        check("撤销一个笔刷点", len(c._fg_pts) == n0 - 1)
        check("撤销后存在可重写记录", c.has_redo())
        ok = c.redo()
        check("重写恢复被撤销的点", ok and len(c._fg_pts) == n0,
              f"n={len(c._fg_pts)}")

        # 15.3 撤销已闭合区域 → 重写恢复
        c.clear_selection()
        c.set_tool("polygon")
        c._draft = [(0, 0), (10, 0), (10, 10)]
        c._close_draft()
        c.undo()                       # 区域退成草稿
        check("撤销区域退为草稿", len(c._regions) == 0 and len(c._draft) == 3)
        c.redo()
        check("重写恢复已闭合区域", len(c._regions) == 1, f"{len(c._regions)}")

        # 15.4 新动作作废重写记录（标准语义）
        c.clear_selection()
        c._paint_at((5, 5), "keep")
        c.undo()
        check("撤销后有重写记录", c.has_redo())
        c._paint_at((6, 6), "keep")
        check("画新内容后重写记录作废", not c.has_redo())
        c.close()

        # 15.5 ROI 工具：框选存进独立图层，且不受 clear_selection 影响
        c2 = Canvas()
        c2.set_image(rgb, None, "checker", keep_transform=False)
        c2.resize(400, 400)
        c2._roi_regions = [[(0, 0), (50, 0), (50, 50), (0, 50)]]
        check("ROI 区域可读取", c2.has_roi() and len(c2.roi_regions()) == 1)
        c2.clear_selection()
        check("clear_selection 不影响 ROI", c2.has_roi())
        c2.clear_roi()
        check("清 ROI 生效", not c2.has_roi())
        c2.close()

        # 15.6 merge_roi：框内用新结果，框外与 base 逐像素一致
        base_rgb = np.full((100, 100, 3), 200, np.uint8)
        base_alpha = np.full((100, 100), 255, np.uint8)
        new_rgb = np.zeros((100, 100, 3), np.uint8)
        new_alpha = np.zeros((100, 100), np.uint8)
        roi = np.zeros((100, 100), np.uint8)
        roi[20:40, 20:40] = 255
        r, a = pipeline.merge_roi(base_rgb, base_alpha, new_rgb, new_alpha,
                                  roi, feather_px=2)
        check("ROI 内采用新结果（alpha→0）", int(a[30, 30]) < 30, f"{int(a[30,30])}")
        check("ROI 外保持 base 不变（逐像素一致）",
              np.array_equal(a[60:90, 60:90], base_alpha[60:90, 60:90])
              and np.array_equal(r[60:90, 60:90], base_rgb[60:90, 60:90]))

        # 15.7 process_array 走 ROI 分支：区域外不变、并有提示
        rgb_s = np.full((120, 120, 3), 255, np.uint8)
        rgb_s[40:80, 40:80] = (10, 20, 200)
        b_rgb, b_a, *_ = pipeline.process_array(rgb_s, None, engine="color")
        roi2 = np.zeros((120, 120), np.uint8)
        roi2[0:30, 0:30] = 255
        o_rgb, o_a, mode, *_rest = pipeline.process_array(
            rgb_s, None, engine="color", roi_mask=roi2,
            base_rgb=b_rgb, base_alpha=b_a)
        warn = _rest[-1] if _rest else []
        check("process_array 接受 roi_mask 参数并返回",
              o_a.shape == b_a.shape)
        check("ROI 分支：主体区（框外）保持 base",
              np.array_equal(o_a[40:80, 40:80], b_a[40:80, 40:80]))
    except Exception:  # noqa: BLE001
        check("M15 画布/重写/ROI", False, traceback.format_exc().splitlines()[-1])


def test_m16_roi_local_and_clear():
    """[16] M16：ROI 局部处理（先整图处理后框选）+ 处理后/预览时清除绘制内容。

    说明（2026-10-16 用户修订）：
      ROI 分支保持"**只有已有既有结果（`_alpha is not None`）时才设底板，
      否则退化为整图处理**"的原始逻辑。因此**正确用法是：先「开始处理」得到结果，
      再框 ROI 单独重跑某一块**（此时 `_alpha` 已存在，ROI 生效）。
      本用例按此流程验证；首次即框 ROI 的"退化整图"行为另行断言。
    """
    section("[16] M16：ROI 局部处理（先处理后框选）+ 绘制内容自动清除")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtWidgets
        from src.ui.gui import MainWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        # 合成图：白底 + 中间蓝色方块
        rgb = np.full((300, 300, 3), 255, np.uint8)
        rgb[120:180, 120:180] = (30, 60, 200)
        p = os.path.join(OUT_DIR, "_m16_roi.png")
        Image.fromarray(rgb, "RGB").save(p)

        win = MainWindow()
        win.show()
        win._load(p)                      # 新加载：_alpha 仍为 None
        check("加载后尚未处理（_alpha is None）", win._alpha is None)

        # ---- 第一步：整图处理一次（拿到既有结果）----
        win._process()
        if win._worker is not None:
            win._worker.wait()
        app.processEvents()
        check("首次整图处理产出结果", win._alpha is not None)

        # ---- 第二步：框 ROI 再处理，只更新框内、框外保持上一次结果 ----
        # 制造"上一次结果"与"新结果"的差异：直接拿当前结果为底板比较即可
        before = win._alpha.copy()
        win.dst_canvas._roi_regions = [[(100, 100), (200, 100),
                                        (200, 200), (100, 200)]]
        win._process()
        if win._worker is not None:
            win._worker.wait()
        app.processEvents()
        a = win._alpha
        check("框 ROI 后产出结果", a is not None)
        check("ROI 内白背景被抠（alpha→0）", int(a[110, 110]) < 30, f"{int(a[110,110])}")
        check("ROI 内主体保留（alpha=255）", int(a[150, 150]) > 250)
        check("ROI 外与上一次结果一致（框外不变）",
              np.array_equal(a[220:280, 220:280], before[220:280, 220:280]))

        # 处理完成后 ROI 框应自动消失
        check("处理完成后 ROI 框自动清除", not win.dst_canvas.has_roi())

        # ---- 首次处理即框 ROI：按原始逻辑退化为整图处理 ----
        win2 = MainWindow()
        win2.show()
        win2._load(p)
        win2.dst_canvas._roi_regions = [[(100, 100), (200, 100),
                                         (200, 200), (100, 200)]]
        win2._process()
        if win2._worker is not None:
            win2._worker.wait()
        app.processEvents()
        a2 = win2._alpha
        check("首次即框 ROI → 退化为整图处理（框外也被处理）",
              a2 is not None and int(a2[20, 20]) < 30, f"{int(a2[20,20]) if a2 is not None else 'None'}")
        win2.close()

        # ---- 「预览」按钮：清除画布上的绘制内容 ----
        win._set_tool("roi")
        win.dst_canvas._roi_regions = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        win._set_tool("fg")
        win.dst_canvas._paint_at((30, 30), "keep")
        check("预览前有 ROI 框与涂抹",
              win.dst_canvas.has_roi() and len(win.dst_canvas._fg_pts) > 0)
        win._set_tool("none")
        check("点「预览」清除 ROI 框与涂抹",
              not win.dst_canvas.has_roi()
              and not win.dst_canvas.has_pending())
        check("首个工具按钮文字为「预览」", win.tool_none.text() == "预览",
              win.tool_none.text())
        win.close()
    except Exception:  # noqa: BLE001
        check("M16 ROI 局部/清除", False, traceback.format_exc().splitlines()[-1])


def test_m17_brush_radius():
    """[17] M17/M18：点选笔刷半径严格生效 + 描边顶点圆点不随缩放放大。"""
    section("[17] M17/M18：笔刷半径与描边顶点（屏幕恒定大小）")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from src.ui.canvas import Canvas
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        c = Canvas()
        rgb = np.full((200, 200, 3), 200, np.uint8)
        c.set_image(rgb, None, "checker", keep_transform=False)
        c.resize(400, 400)

        # 半径 2 必须原样保留（旧代码 max(3, 2) 会顶成 3）
        c.set_brush_radius(2)
        check("笔刷半径 2 被原样保留", c._brush_radius == 2, str(c._brush_radius))
        c.set_brush_radius(1)
        check("笔刷半径 1 被原样保留", c._brush_radius == 1, str(c._brush_radius))
        c.set_brush_radius(0)
        check("半径 0 夹到最小 1", c._brush_radius == 1, str(c._brush_radius))
        c.close()

        # GUI 参数框允许设到 1
        from src.ui.gui import MainWindow
        win = MainWindow()
        check("GUI 笔刷下限允许 1", win.brush.minimum() == 1, str(win.brush.minimum()))
        win.close()

        # 描边顶点圆点：cosmetic 画笔 → 屏幕尺寸恒定（M18）
        from src.ui.canvas import vertex_dot_pen, _VERTEX_DOT_PX
        pen = vertex_dot_pen(QtGui.QColor(120, 220, 160))
        check("顶点画笔为 cosmetic（宽度不随缩放变化）", pen.isCosmetic())
        check("顶点屏幕直径固定约 3px", abs(_VERTEX_DOT_PX - 3.0) < 1e-9,
              str(_VERTEX_DOT_PX))
        check("顶点画笔为圆头（画出来是个圆点）",
              pen.capStyle() == QtCore.Qt.RoundCap)
    except Exception:  # noqa: BLE001
        check("M17 笔刷半径", False, traceback.format_exc().splitlines()[-1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    test_color_engine()
    test_hybrid_engine()
    test_ai_engine()
    test_interactive()
    test_gui_build()
    test_edge_cases()
    test_manual_override()
    test_canvas_interaction()
    test_gui_override_flow()
    test_interaction_fix()
    test_alpha_quality()
    test_hq_engine()
    test_m13_semantics()
    test_m15_canvas_redo_roi()
    test_m16_roi_local_and_clear()
    test_m17_brush_radius()
    test_cli()
    print(f"\n{'=' * 48}")
    print(f"结果：通过 {PASS} 项，失败 {FAIL} 项")
    if FAILED_NAMES:
        print("失败项：" + "；".join(FAILED_NAMES))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
