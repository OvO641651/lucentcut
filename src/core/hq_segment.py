"""高质量抠图引擎（M12）：BiRefNet / ISNet 离线推理。

背景与动机
----------
M1–M11 的复杂背景路径用的是内置 u2netp（约 4.5MB，输入固定 320×320）。
它够小、够快、能跑在任意机器上，但 320×320 的输入分辨率决定了它**看不清
细轮廓**：人物头发会被糊成一个圆润的团块，半透明发丝、飘带、细长道具
（剑、发饰）都会丢失。实测 image/1.jpg（动漫四人合影，渐变天空背景）用
u2netp 抠出来的轮廓是"圆头团块"，与用户用其他软件得到的结果差距明显。

本模块引入**语义分割质量更高的大模型**作为可选引擎：

- BiRefNet（Zheng et al., "Bilateral Reference for High-Resolution
  Dichotomous Image Segmentation"，MIT 许可）：输入 1024×1024，
  官方给出的对比里"模糊边缘像素"占比 0.86%，而 U2Net 是 2.56%（约 3 倍更利落）。
- ISNet general-use：rembg 里常规照片场景的默认高质量模型，1024×1024。

设计原则（与 ai_segment 一致）
-----------------------------
- **纯离线**：只读本地 onnx 文件，不联网、不上传图片。
- **自包含**：只依赖 onnxruntime + numpy + Pillow，不 import rembg
  （rembg 顶层 import scipy/pymatting/skimage，打包易失效，M3 已踩过坑）。
- **可缺省**：模型文件不存在时 is_available() 返回 False，调用方自动回退
  到 u2netp，绝不因此报错。
- **预处理/后处理逐像素对齐 rembg**：normalize 用的是 rembg
  `BaseSession.normalize` 的同一套公式（RGB → LANCZOS resize 到模型输入
  尺寸 → 除以全图最大值 → 按通道 (x-mean)/std → CHW），后处理同样是
  sigmoid（BiRefNet 需要）→ 全图 min-max 归一化 → resize 回原尺寸。

模型目录约定
------------
    <base>/models/models/birefnet/birefnet_tiny.onnx
    <base>/models/models/isnet/isnet.onnx

可用 tools/download_models.py 下载。放入更大的模型（如 2560×1440 的
BiRefNet lite-2K）也能直接识别：输入尺寸从 onnx 图里读，不写死。
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# 归一化常量（与 rembg 一致）
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_HALF_MEAN = (0.5, 0.5, 0.5)
_HALF_STD = (1.0, 1.0, 1.0)


class HqSegmentError(RuntimeError):
    """高质量抠图引擎不可用或推理失败。"""


@dataclass
class HqResult:
    rgb: np.ndarray                 # 前景 RGB（透明区已做边缘色外扩，防黑边）
    alpha: np.ndarray               # uint8, (H, W)
    model_name: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelSpec:
    """一个可选模型的描述。patterns 按优先级排列（先匹配到的先用）。"""

    key: str
    label: str
    patterns: Tuple[str, ...]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    sigmoid: bool                   # 输出是否需要过 sigmoid（BiRefNet 需要）


# 注册表：顺序即优先级（大模型优先，质量更好；缺了自动往下找）
MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="birefnet",
        label="BiRefNet",
        patterns=("birefnet/birefnet_tiny.onnx", "birefnet/*.onnx", "**/birefnet*.onnx"),
        mean=_IMAGENET_MEAN, std=_IMAGENET_STD, sigmoid=True,
    ),
    ModelSpec(
        key="isnet",
        label="ISNet",
        patterns=("isnet/isnet.onnx", "isnet/*.onnx", "**/isnet*.onnx"),
        mean=_HALF_MEAN, std=_HALF_STD, sigmoid=False,
    ),
)


# --------------------------------------------------------------------------- #
# 路径与模型发现
# --------------------------------------------------------------------------- #
def _base_dir() -> str:
    """项目/包根目录，兼容源码运行与 PyInstaller 打包运行。"""
    if getattr(sys, "_MEIPASS", None):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model_roots() -> List[str]:
    """按优先级排列的模型搜索根目录。"""
    base = _base_dir()
    roots = [
        os.path.join(base, "models"),              # <base>/models（U2NET_HOME 结构）
        os.path.join(base, "models", "models"),
    ]
    extra = os.environ.get("LUCENTCUT_MODEL_ROOT")
    if extra:
        roots.insert(0, extra)
    return roots


def find_model() -> Optional[Tuple[ModelSpec, str]]:
    """返回 (spec, onnx 绝对路径)；找不到任何高质量模型时返回 None。"""
    for spec in MODEL_SPECS:
        for root in _model_roots():
            for pat in spec.patterns:
                hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))
                # 跳过未下完的断点文件
                hits = [h for h in hits if not h.endswith(".part")]
                if hits:
                    return spec, hits[0]
    return None


def availability_detail() -> Tuple[bool, str]:
    """返回 (是否可用, 说明)。界面用它给出精确原因。"""
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"缺少 onnxruntime（{type(exc).__name__}）"
    hit = find_model()
    if not hit:
        return False, ("未找到高质量模型（可运行 tools/download_models.py "
                       "下载 BiRefNet，或直接用 u2netp 引擎）")
    spec, path = hit
    return True, f"{spec.label} 就绪（{os.path.basename(path)}）"


def is_available() -> bool:
    return availability_detail()[0]


def model_label() -> str:
    hit = find_model()
    return hit[0].label if hit else ""


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
_session = None
_session_path: Optional[str] = None
_session_spec: Optional[ModelSpec] = None
_session_input_hw: Optional[Tuple[int, int]] = None   # (H, W)
_session_input_name: Optional[str] = None


def _input_size(sess) -> Tuple[int, int]:
    """从 onnx 图里读模型输入尺寸，返回 (H, W)。

    固定尺寸（int）直接用；动态维度（字符串/None）退化为 1024×1024。
    这样换成 2560×1440 的 lite-2K 模型也不用改代码。
    """
    shape = sess.get_inputs()[0].shape      # 例如 [1, 3, 1024, 1024]
    h, w = 1024, 1024
    if len(shape) == 4:
        th, tw = shape[2], shape[3]
        if isinstance(th, int) and th > 0:
            h = th
        if isinstance(tw, int) and tw > 0:
            w = tw
    return h, w


def _get_session():
    """懒加载并缓存 onnxruntime Session（含模型规格、输入尺寸与输入名）。"""
    global _session, _session_path, _session_spec
    global _session_input_hw, _session_input_name
    hit = find_model()
    if not hit:
        raise HqSegmentError("未找到高质量模型文件，无法推理")
    spec, path = hit
    if _session is not None and _session_path == path:
        return _session, spec, _session_input_hw, _session_input_name

    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        raise HqSegmentError(f"onnxruntime 不可用：{exc}") from exc

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0            # 交给 onnxruntime 自动选线程数
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 关键（2026-10-11 实测）：BiRefNet 在 1024×1024 下单次推理的中间张量
        # 很大，默认的内存池会把每一轮推理的峰值一直占住 —— 同一进程里连跑
        # 第二张图就会抛 "bad allocation"。关掉 mem_pattern 与 cpu_mem_arena
        # 后，逐轮内存可完全回收（实测连跑 4 轮可用内存稳定在 8.2GB 不下降）。
        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = False
        sess = ort.InferenceSession(path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        raise HqSegmentError(f"高质量模型加载失败：{exc}") from exc

    _session = sess
    _session_path = path
    _session_spec = spec
    _session_input_hw = _input_size(sess)
    _session_input_name = sess.get_inputs()[0].name
    return _session, spec, _session_input_hw, _session_input_name


def _normalize(rgb: np.ndarray, size_hw: Tuple[int, int],
               mean, std) -> np.ndarray:
    """与 rembg BaseSession.normalize 逐像素一致的预处理。

    返回 float32 的 (1, 3, H, W)。
    """
    from PIL import Image

    h, w = size_hw
    im = Image.fromarray(rgb, mode="RGB").resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float64)
    arr = arr / max(float(arr.max()), 1e-6)
    out = np.empty((h, w, 3), dtype=np.float64)
    for c in range(3):
        out[:, :, c] = (arr[:, :, c] - mean[c]) / std[c]
    return out.transpose(2, 0, 1)[None].astype(np.float32)


def _postprocess(pred: np.ndarray, size_hw: Tuple[int, int],
                 sigmoid: bool) -> np.ndarray:
    """模型输出 → 与原图同尺寸的 uint8 alpha（对齐 rembg 后处理）。"""
    from PIL import Image

    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim == 4:
        pred = pred[:, 0, :, :]
    pred = np.squeeze(pred)

    # 输出是 logits 还是概率，以实际数值为准（比写死更稳）
    if sigmoid and (pred.min() < -1e-3 or pred.max() > 1.0 + 1e-3):
        pred = 1.0 / (1.0 + np.exp(-pred))

    mi, ma = float(pred.min()), float(pred.max())
    if ma - mi < 1e-6:
        pred = np.zeros_like(pred)
    else:
        pred = (pred - mi) / (ma - mi)

    h, w = size_hw
    mask = Image.fromarray((pred * 255.0).astype(np.uint8), mode="L")
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.Resampling.LANCZOS)
    return np.asarray(mask, dtype=np.uint8)


def predict_alpha(rgb: np.ndarray) -> np.ndarray:
    """高质量抠图：返回与原图同尺寸的 alpha（uint8, H×W）。"""
    if not is_available():
        _, why = availability_detail()
        raise HqSegmentError(f"高质量抠图不可用：{why}")

    sess, spec, (in_h, in_w), iname = _get_session()
    x = _normalize(rgb, (in_h, in_w), spec.mean, spec.std)
    try:
        outs = sess.run(None, {iname: x})
    except Exception as exc:  # noqa: BLE001
        raise HqSegmentError(f"高质量模型推理失败：{exc}") from exc

    h, w = rgb.shape[:2]
    return _postprocess(outs[0], (h, w), spec.sigmoid)


def remove_background(rgb: np.ndarray) -> HqResult:
    """高质量抠图，返回前景（透明区 RGB 已做边缘外扩，防黑边）。"""
    from . import refine

    alpha = predict_alpha(rgb)
    rgb_out = refine.extend_edge_color(rgb, alpha)
    return HqResult(rgb=rgb_out, alpha=alpha, model_name=model_label())


def rescue_subject(color_alpha: np.ndarray, hq_alpha: np.ndarray,
                   hi: int = 170, lo: int = 128,
                   min_area_ratio: float = 0.0015) -> Tuple[np.ndarray, float]:
    """用高质量模型的主体掩码补齐颜色键控误删的同色部件。

    与 ai_segment.rescue_subject 同语义（见该函数注释），只是把 u2netp
    的粗糙掩码换成 BiRefNet 的精细掩码，白衣服边界更准、毛边更少。
    """
    from . import colorkey
    from .ai_segment import _keep_large_components

    subj = _keep_large_components(hq_alpha >= hi, min_area_ratio)
    restore = subj & (color_alpha < lo)
    if not restore.any():
        return color_alpha, 0.0

    out = color_alpha.copy()
    out[restore] = 255
    out = colorkey.keep_border_connected(out)
    return out, float(restore.mean())
