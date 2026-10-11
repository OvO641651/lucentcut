"""LucentCut 命令行入口（临时自测用，正式 GUI 见 M2）。

用法示例
--------
不带参数直接运行（在 PyCharm 里右键 Run 最方便）：
    python -m src.main
    → 弹出文件选择窗口挑图片，处理结果自动保存到原图所在目录的 output/ 下

单张处理（自动识别背景，默认精细抠图引擎）：
    python -m src.main -i image/deepseek.jpg -o output/deepseek_nobg.png

复杂背景想要最好的头发/细轮廓（慢一点，单张约 8~9 秒）：
    python -m src.main -i image/1.jpg -o output/1_nobg.png --engine matting

追求速度、不要大模型：
    python -m src.main -i image/deepseek.jpg -o out.png --engine hybrid

批量处理整个目录：
    python -m src.main -i image/ -o output/ --batch

指定白底模式并调整灵敏度：
    python -m src.main -i input.png -o out.png --mode white --t-low 25 --t-high 70

禁用"沿用已有 alpha"（强制按颜色重抠）：
    python -m src.main -i a.png -o b.png --no-reuse-alpha
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from .core import pipeline
except ImportError:  # 允许直接 python src/main.py 运行
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.core import pipeline

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ask_files_via_dialog() -> list[str]:
    """无命令行参数时弹出文件选择窗口，返回选中的图片路径列表。

    tkinter 是 Python 标准库自带组件。本项目一期不引入 PySide6，
    这里仅作为「不带参数直接运行时的兜底交互」，正式桌面 GUI 见 M2。
    """
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="选择要处理的图片（可多选）",
        filetypes=[("图片文件", " ".join(f"*{e}" for e in IMAGE_EXT)), ("所有文件", "*.*")],
    )
    root.destroy()
    return list(paths)


def print_usage_hint() -> None:
    """没有图形界面时，给出可直接复制的命令行示例。"""
    print("未指定输入文件。请用 -i 指定图片或目录，例如：\n", file=sys.stderr)
    print("    python -m src.main -i image/deepseek.jpg -o output/out.png", file=sys.stderr)
    print("    python -m src.main -i image/ -o output/ --batch\n", file=sys.stderr)
    print("如需查看全部参数：python -m src.main -h", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lucentcut",
        description="去除图片白底/黑底背景，输出透明 PNG",
    )
    p.add_argument("-i", "--input", default=None,
                   help="输入图片或目录；省略则弹出窗口选择")
    p.add_argument("-o", "--output", default=None, help="output")
    p.add_argument("--batch", action="store_true", help="批量处理目录")
    p.add_argument("--mode", default="auto", choices=["auto", "white", "black", "solid"],
                   help="背景模式，默认 auto 自动识别")
    p.add_argument("--t-low", type=float, default=30.0, help="软阈值下界（默认 30）")
    p.add_argument("--t-high", type=float, default=80.0, help="软阈值上界（默认 80）")
    p.add_argument("--feather", type=int, default=None, help="羽化半径（默认按尺寸自适应）")
    p.add_argument("--erode", type=int, default=0, help="alpha 收缩像素（默认 0）")
    p.add_argument("--denoise", type=int, default=0, help="去噪：小于该面积的前景块丢弃（默认关闭）")
    p.add_argument("--no-despill", action="store_true", help="关闭去白边")
    p.add_argument("--no-fix-holes", action="store_true", help="关闭主体内部连通域修正")
    p.add_argument("--no-reuse-alpha", action="store_true",
                   help="检测到输入已有透明通道时仍按颜色重抠")
    p.add_argument("--engine", default="matting",
                   choices=["matting", "hybrid", "color", "ai"],
                   help="处理引擎：matting=精细抠图(默认，复杂背景走 BiRefNet 大模型)，"
                        "hybrid=颜色轮廓+AI补齐(快)，color=纯颜色键控，ai=纯 AI 分割")
    return p


def _kwargs_from_args(args) -> dict:
    return dict(
        mode=args.mode,
        t_low=args.t_low,
        t_high=args.t_high,
        feather_radius=args.feather,
        erode_px=args.erode,
        despill_on=not args.no_despill,
        denoise_min_area=args.denoise,
        fix_holes=not args.no_fix_holes,
        reuse_existing_alpha=not args.no_reuse_alpha,
        engine=args.engine,
    )


def _print_single(res) -> None:
    print(f"模式      : {res.mode}"
          + ("（沿用输入自带的透明通道）" if res.reused_existing_alpha else ""))
    print(f"尺寸      : {res.width} x {res.height}")
    print(f"背景色估计: {res.bg_color}")
    print(f"背景透明率: {res.bg_transparent_ratio:.1%}")
    print(f"主体保留率: {res.fg_keep_ratio:.1%}")
    if res.rescued_ratio > 0:
        print(f"AI 补齐率 : {res.rescued_ratio:.1%}")
    print(f"输出      : {res.output_path}")
    for w in res.warnings:
        print(f"提示      : {w}")


def _run_interactive(kwargs: dict) -> int:
    """没有 -i 时的交互路径：弹窗选图，逐张处理。"""
    try:
        files = ask_files_via_dialog()
    except Exception:  # noqa: BLE001  无图形界面环境（如纯终端）
        print_usage_hint()
        return 2

    if not files:
        print("未选择任何文件，已取消。")
        return 1

    ok = 0
    for idx, src in enumerate(files, 1):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(src)), "output")
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(src))[0] + "_nobg.png"
        print(f"\n[{idx}/{len(files)}] {os.path.basename(src)}")
        try:
            res = pipeline.process_image(src, os.path.join(out_dir, name), **kwargs)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  跳过：{exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  失败：{type(exc).__name__}: {exc}")
            continue
        _print_single(res)
        ok += 1

    print(f"\n完成：成功 {ok} 张 / 共 {len(files)} 张，输出目录 output/")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = _kwargs_from_args(args)

    # 未指定输入 → 走交互（弹窗选图）路径
    if not args.input:
        return _run_interactive(kwargs)

    if args.batch or os.path.isdir(args.input):
        out_dir = args.output or "output"
        results = pipeline.process_batch(args.input, out_dir, **kwargs)
        ok = sum(1 for r in results if r.mode != "skipped")
        print(f"{'文件':<24}{'模式':<9}{'背景透明':<10}{'主体保留':<10}说明")
        print("-" * 84)
        for r in results:
            note = "；".join(r.warnings) if r.warnings else "OK"
            if r.mode == "skipped":
                print(f"{os.path.basename(r.input_path):<24}{'跳过':<9}{'-':<10}{'-':<10}{note}")
            else:
                print(f"{os.path.basename(r.input_path):<24}{r.mode:<9}"
                      f"{r.bg_transparent_ratio:<10.1%}{r.fg_keep_ratio:<10.1%}{note}")
        print(f"\n完成：成功 {ok} 张，跳过 {len(results) - ok} 张，输出目录 {out_dir}")
        return 0

    try:
        res = pipeline.process_image(args.input, args.output, **kwargs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1

    _print_single(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
