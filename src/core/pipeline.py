"""单图处理流程串联（M1，M8 起支持混合引擎）。

load_image → 【已有干净 alpha 则直接沿用】→ 背景识别 → 纯色抠图 / AI 分割 / 混合
          → 边缘优化 → alpha 合并 → 保存。

引擎（engine 参数）
------------------
- "matting"（M12 新增，GUI 默认）：精细抠图。复杂背景走 BiRefNet / ISNet
  大模型（1024×1024，头发丝/飘带/细长道具都留得住，单张约 8~9 秒）；
  纯色背景自动落回颜色键控轮廓（更快且边缘更贴合）。模型缺失或推理失败时
  自动回退到 u2netp / 颜色键控，不会因此报错。
- "hybrid"：颜色键控取轮廓 + AI 补齐被误删的白色/同色主体部件。
  AI 不可用时自动退化为纯颜色键控。复杂背景自动走纯 AI（u2netp，320×320）。
- "color" ：纯颜色键控（快，适合干净的白底/黑底图）。
- "ai"    ：纯 AI 语义分割（u2netp，复杂背景、照片类）。

关于"沿用已有 alpha"（技术方案 5.1 Step 0，2026-10-10 实测补充）：
    已经抠好的 PNG，其透明区下方 RGB 通常被压成纯黑。若不加判断地按颜色
    再抠一遍，算法会把背景识别成"黑底"，进而删掉主体上偏暗的部分（实测
    deepseek4.png 误删 2.7% 主体像素，均为亮度约 16 的深色区域）。
    因此凡检测到干净完整的 alpha，直接沿用，跳过颜色键控。

关于人工裁决层（M9，决策 D9/D10）：
    用户手动描出的"框内保留 / 框内去除"不与算法结果混在一起计算，而是作为
    **独立的最终裁决层**保存在主窗口里，每次处理完成后叠加应用。因此无论
    先处理再描边、还是描边后再点一次"开始处理"，裁决都不会丢失，且始终
    作用于**最新一次**算法结果。裁决优先于算法；保留区若原被算法判成背景，
    用原图 RGB 回填颜色（见 interactive.apply_override）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import ai_segment, colorkey, hq_segment, interactive, io, refine


@dataclass
class ProcessResult:
    input_path: str
    output_path: str
    mode: str
    width: int
    height: int
    bg_color: tuple
    bg_transparent_ratio: float   # 背景（边框）区域透明像素占比
    fg_keep_ratio: float          # 前景保留像素占比
    reused_existing_alpha: bool = False   # 是否直接沿用了输入自带的 alpha
    rescued_ratio: float = 0.0            # 混合引擎补回的像素占比
    warnings: List[str] = field(default_factory=list)


def auto_feather_radius(height: int, width: int) -> int:
    """按图片尺寸自适应羽化半径。

    极小图标（如 30×35）若仍按 1px 羽化，边缘过渡带会占掉轮廓的很大比例，
    看起来发虚；大图则不受影响，保持 1px 平滑过渡。
    """
    short = min(height, width)
    if short <= 64:
        return 0
    return 1


# --------------------------------------------------------------------------- #
# 核心：对数组做处理（GUI 与 CLI 共用，避免两套逻辑不一致）
# --------------------------------------------------------------------------- #
def process_array(
    rgb: np.ndarray,
    origin_alpha: Optional[np.ndarray] = None,
    mode: str = "auto",
    t_low: float = 30.0,
    t_high: float = 80.0,
    feather_radius: Optional[int] = None,   # None = 按尺寸自适应
    erode_px: int = 0,
    despill_on: bool = True,
    denoise_min_area: int = 0,
    fix_holes: bool = True,
    reuse_existing_alpha: bool = True,
    harden_existing_alpha: bool = True,
    engine: str = "hybrid",   # hybrid / color / ai
    keep_mask: Optional[np.ndarray] = None,   # 人工裁决：必须保留
    drop_mask: Optional[np.ndarray] = None,   # 人工裁决：必须去除
    override_feather: int = 1,                # 裁决层边界羽化像素
) -> Tuple[np.ndarray, np.ndarray, str, float, bool, List[str]]:
    """处理一张图（含人工裁决层叠加）。

    返回 (rgb_out, alpha_out, used_mode, rescued_ratio, reused, warnings)。
    人工裁决层在此处统一叠加，因此"再点一次开始处理"不会丢失用户修改。
    """
    rgb_out, alpha_out, used_mode, rescued, reused, warnings = _process_core(
        rgb, origin_alpha, mode=mode, t_low=t_low, t_high=t_high,
        feather_radius=feather_radius, erode_px=erode_px, despill_on=despill_on,
        denoise_min_area=denoise_min_area, fix_holes=fix_holes,
        reuse_existing_alpha=reuse_existing_alpha,
        harden_existing_alpha=harden_existing_alpha, engine=engine)

    if keep_mask is not None or drop_mask is not None:
        rgb_out, alpha_out = interactive.apply_override(
            rgb_out, alpha_out, rgb, keep_mask, drop_mask,
            feather_px=override_feather)
        n_keep = int((keep_mask > 0).sum()) if keep_mask is not None else 0
        n_drop = int((drop_mask > 0).sum()) if drop_mask is not None else 0
        if n_keep or n_drop:
            warnings.append(
                f"已叠加人工修改：保留 {n_keep} px、去除 {n_drop} px（优先于算法）")
    return rgb_out, alpha_out, used_mode, rescued, reused, warnings


def _process_core(
    rgb: np.ndarray,
    origin_alpha: Optional[np.ndarray] = None,
    mode: str = "auto",
    t_low: float = 30.0,
    t_high: float = 80.0,
    feather_radius: Optional[int] = None,   # None = 按尺寸自适应
    erode_px: int = 0,
    despill_on: bool = True,
    denoise_min_area: int = 0,
    fix_holes: bool = True,
    reuse_existing_alpha: bool = True,
    harden_existing_alpha: bool = True,
    engine: str = "hybrid",   # hybrid / color / ai
) -> Tuple[np.ndarray, np.ndarray, str, float, bool, List[str]]:
    """纯算法处理（不含人工裁决层）。返回签名同 process_array。"""
    warnings: List[str] = []
    h, w = rgb.shape[:2]
    if feather_radius is None:
        feather_radius = auto_feather_radius(h, w)

    reused = False
    trimmed = False
    rescued = 0.0

    # ---------- Step 0：输入是否已自带干净的透明通道 ----------
    trimmed_info: Optional[dict] = None
    if reuse_existing_alpha and origin_alpha is not None:
        trimmed, trimmed_info = io.alpha_is_trimmed(origin_alpha)

    if trimmed:
        reused = True
        alpha = origin_alpha.copy()
        used_mode = "reuse"
        bg = colorkey.estimate_background(rgb)
        rgb_out, alpha_out = refine.refine(
            rgb, alpha, bg,
            feather_radius=0, erode_px=0, despill_on=False, denoise_min_area=0,
        )
        # 沿用模式下不动可见像素：alpha 硬化（254→255）+ 边缘色外扩（防黑边）
        if harden_existing_alpha:
            alpha_out = refine.harden_alpha(alpha_out)
        rgb_out = refine.extend_edge_color(rgb_out, alpha_out)
        warnings.append(
            f"输入已含干净透明通道（二值占比 {trimmed_info['binary_ratio']:.1%}、"
            f"边框透明率 {trimmed_info['border_transparent']:.1%}），直接沿用未重抠"
        )
        return rgb_out, alpha_out, used_mode, rescued, reused, warnings

    if origin_alpha is not None and trimmed_info is not None:
        warnings.append(
            f"输入含 alpha 但未判定为已抠好：{trimmed_info['reason']}；"
            "已按所选引擎处理并与原 alpha 合并"
        )

    # ---------- 纯 AI ----------
    if engine == "ai":
        ai = ai_segment.remove_background(rgb)
        alpha_out = refine.harden_alpha(ai.alpha)
        if denoise_min_area > 0:
            alpha_out = refine.denoise(alpha_out, denoise_min_area)
        if origin_alpha is not None:
            alpha_out = refine.merge_alpha(alpha_out, origin_alpha)
        warnings.append("已用 AI 语义分割提取主体")
        return ai.rgb, alpha_out, "ai", 0.0, reused, warnings

    is_complex = (mode == "auto" and colorkey.detect_mode(rgb) == colorkey.COMPLEX) \
        or mode == colorkey.COMPLEX

    # ---------- 精细抠图（M12：BiRefNet / ISNet 大模型） ----------
    # 只在"复杂背景"时启用大模型：它吃分辨率（1024×1024），头发丝/飘带/细长道具
    # 都能留住；代价是单张约 8~9 秒（CPU）。纯色背景不受益 —— 颜色键控的轮廓比
    # 任何语义分割都更贴合原始像素，所以那种情况自动落回下面的混合路径。
    if engine == "matting":
        hq_ok, hq_why = hq_segment.availability_detail()
        if not hq_ok:
            warnings.append(f"精细抠图模型不可用（{hq_why}），已回退到 AI/颜色键控")
        elif is_complex:
            try:
                hq = hq_segment.remove_background(rgb)
                alpha_out = refine.harden_alpha(hq.alpha)
                if denoise_min_area > 0:
                    alpha_out = refine.denoise(alpha_out, denoise_min_area)
                if origin_alpha is not None:
                    alpha_out = refine.merge_alpha(alpha_out, origin_alpha)
                warnings.append(
                    f"复杂背景，已用 {hq.model_name} 精细抠图"
                    "（大模型推理较慢，单张约 8~9 秒）")
                return hq.rgb, alpha_out, "matting", 0.0, reused, warnings
            except Exception as exc:  # noqa: BLE001  模型异常不应让整张图失败
                warnings.append(
                    f"精细抠图失败（{type(exc).__name__}: {exc}），已回退到 AI 分割")
        else:
            warnings.append("纯色背景：精细引擎自动采用颜色键控轮廓（与混合引擎一致）")

    # ---------- 混合引擎 ----------
    if engine in ("hybrid", "matting") and ai_segment.is_available():
        if is_complex:
            ai = ai_segment.remove_background(rgb)
            alpha_out = refine.harden_alpha(ai.alpha)
            if denoise_min_area > 0:
                alpha_out = refine.denoise(alpha_out, denoise_min_area)
            if origin_alpha is not None:
                alpha_out = refine.merge_alpha(alpha_out, origin_alpha)
            warnings.append("复杂背景，已自动用 AI 语义分割提取主体")
            return ai.rgb, alpha_out, "ai", 0.0, reused, warnings

        alpha, used_mode, bg = colorkey.remove_solid_bg(
            rgb, mode=mode, t_low=t_low, t_high=t_high, fix_holes=fix_holes)
        # 颜色键控轮廓 + AI 补齐同色主体部件（白色衣物/头饰等）
        ai_alpha = ai_segment.predict_alpha(rgb)
        alpha, rescued = ai_segment.rescue_subject(alpha, ai_alpha)
        if rescued > 0:
            warnings.append(f"AI 补齐同色主体部件（{rescued:.1%} 像素）")
            if used_mode in (colorkey.WHITE, colorkey.BLACK, colorkey.SOLID):
                used_mode = "hybrid-" + used_mode
            else:
                used_mode = "hybrid"

        rgb_out, alpha_out = refine.refine(
            rgb, alpha, bg, feather_radius=feather_radius, erode_px=erode_px,
            despill_on=despill_on, denoise_min_area=denoise_min_area)
        if origin_alpha is not None:
            alpha_out = refine.merge_alpha(alpha_out, origin_alpha)
        return rgb_out, alpha_out, used_mode, rescued, reused, warnings

    # ---------- 纯颜色键控（含复杂背景回退 AI） ----------
    try:
        alpha, used_mode, bg = colorkey.remove_solid_bg(
            rgb, mode=mode, t_low=t_low, t_high=t_high, fix_holes=fix_holes)
    except ValueError as exc:
        try:
            ai = ai_segment.remove_background(rgb)
            alpha_out = refine.harden_alpha(ai.alpha)
            if denoise_min_area > 0:
                alpha_out = refine.denoise(alpha_out, denoise_min_area)
            if origin_alpha is not None:
                alpha_out = refine.merge_alpha(alpha_out, origin_alpha)
            warnings.append("复杂背景，已自动用 AI 语义分割提取主体")
            return ai.rgb, alpha_out, "ai", 0.0, reused, warnings
        except Exception as ai_exc:  # noqa: BLE001
            raise ValueError(f"{exc}；且 AI 分割不可用：{ai_exc}") from exc

    rgb_out, alpha_out = refine.refine(
        rgb, alpha, bg, feather_radius=feather_radius, erode_px=erode_px,
        despill_on=despill_on, denoise_min_area=denoise_min_area)
    if origin_alpha is not None:
        alpha_out = refine.merge_alpha(alpha_out, origin_alpha)
    return rgb_out, alpha_out, used_mode, 0.0, reused, warnings


def process_image(
    input_path: str,
    output_path: Optional[str] = None,
    **kwargs,
) -> ProcessResult:
    """处理单张图片：去背景并保存透明 PNG。kwargs 见 process_array。"""
    rgb, origin_alpha = io.load_image(input_path)

    if output_path is None:
        output_path = io.default_output_path(input_path)

    rgb_out, alpha_out, used_mode, rescued, reused, warnings = process_array(
        rgb, origin_alpha, **kwargs)

    io.save_rgba(rgb_out, alpha_out, output_path)
    return _build_result(input_path, output_path, used_mode,
                         colorkey.estimate_background(rgb), alpha_out,
                         reused, rescued, warnings)


def _build_result(input_path, output_path, used_mode, bg, alpha_out,
                  reused, rescued, warnings):
    h, w = alpha_out.shape
    t = max(1, min(3, h // 2, w // 2))
    border_mask = np.zeros_like(alpha_out, dtype=bool)
    border_mask[:t, :] = border_mask[-t:, :] = True
    border_mask[:, :t] = border_mask[:, -t:] = True
    bg_ratio = float(np.mean(alpha_out[border_mask] < 16))

    if bg_ratio < 0.5 and used_mode != "reuse":
        warnings.append("边框区域透明比例偏低，背景可能未完全去除")

    return ProcessResult(
        input_path=input_path,
        output_path=output_path,
        mode=used_mode,
        width=w,
        height=h,
        bg_color=tuple(int(c) for c in bg),
        bg_transparent_ratio=bg_ratio,
        fg_keep_ratio=float(np.mean(alpha_out >= 128)),
        reused_existing_alpha=reused,
        rescued_ratio=rescued,
        warnings=warnings,
    )


def process_batch(
    input_dir: str,
    output_dir: str,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    progress_cb=None,
    **kwargs,
) -> List[ProcessResult]:
    """批量处理目录下所有图片。

    progress_cb(cur, total, filename)：每处理完一张回调一次（M13 新增，
    供界面进度条使用）。回调抛异常不影响批处理本身。
    """
    os.makedirs(output_dir, exist_ok=True)
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
            files.append((name, path))
    results: List[ProcessResult] = []
    total = len(files)
    for i, (name, path) in enumerate(files):
        out = io.default_output_path(path, output_dir=output_dir)
        try:
            results.append(process_image(path, out, **kwargs))
        except ValueError as exc:
            results.append(ProcessResult(
                input_path=path, output_path="", mode="skipped",
                width=0, height=0, bg_color=(0, 0, 0),
                bg_transparent_ratio=0.0, fg_keep_ratio=0.0,
                warnings=[str(exc)],
            ))
        if progress_cb is not None:
            try:
                progress_cb(i + 1, total, name)
            except Exception:  # noqa: BLE001  进度回调不得中断批处理
                pass
    return results


__all__ = [
    "ProcessResult", "process_array", "process_image", "process_batch",
    "auto_feather_radius", "interactive",
]
