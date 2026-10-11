"""AI 语义分割（M5）：用本地 u2netp 模型离线提取主体，处理复杂背景与白色部件场景。

技术路线（见技术方案 5.2）：
- 不连任何云端 API，模型完全本地推理，图片不出本机。
- 内置 u2netp（轻量 U²-Net，约 4.5MB），适配无独显 / 低显存环境。
- **自包含推理**：直接用 onnxruntime 跑本地 onnx 文件，不再依赖 rembg。
  原因（2026-10-11 修复）：rembg 2.0 的 `bg.py` 在导入时即 `import scipy.ndimage /
  pymatting / skimage`，一旦打包时排除 scipy（或用户环境没装全），`import rembg`
  就会失败，导致"AI 不可用"。改为自实现后，依赖只剩 onnxruntime + numpy + Pillow。
  自实现的前后处理与 rembg u2netp 逐像素完全一致（已实测 max diff = 0）。

对外接口：
- is_available()            运行环境是否就绪（onnxruntime 可导入且模型文件存在）
- availability_detail()     返回 (是否可用, 说明文本)，便于界面提示
- remove_background(rgb)    AI 抠图，返回 AiResult（前景 RGB + alpha）
- rescue_subject(...)       用 AI 主体掩码补齐"颜色键控误删的白色衣物"（混合引擎用）
- AiSegmentError            异常类型
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# u2netp 固定输入尺寸与归一化参数（与 rembg 官方实现保持一致）
_INPUT_SIZE = (320, 320)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_MODEL_FILENAME = "u2netp.onnx"


class AiSegmentError(RuntimeError):
    """AI 分割不可用或失败的异常。"""


@dataclass
class AiResult:
    rgb: np.ndarray               # 前景 RGB（透明区已做边缘色外扩，防黑边）
    alpha: np.ndarray             # uint8, (H, W)
    model_name: str = "u2netp"
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 路径与依赖探测
# --------------------------------------------------------------------------- #
def _base_dir() -> str:
    """返回项目/包的根目录，兼容源码运行与 PyInstaller 打包运行。"""
    if getattr(sys, "_MEIPASS", None):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate_model_dirs() -> List[str]:
    """返回按优先级排列的模型搜索目录。"""
    base = _base_dir()
    candidates = [
        os.path.join(base, "models"),            # <base>/models（U2NET_HOME 结构）
        os.path.join(base, "models", "models"),  # <base>/models/models/<name>/
        os.path.join(os.path.expanduser("~"), ".u2net"),          # 旧版 rembg 默认
        os.path.join(os.path.expanduser("~"), ".rembg", "models"),
    ]
    if os.environ.get("U2NET_HOME"):
        candidates.insert(0, os.environ["U2NET_HOME"])
    return candidates


def _find_model() -> Optional[str]:
    """在候选目录里查找 u2netp.onnx，返回第一个存在的路径。"""
    for d in _candidate_model_dirs():
        for pat in (_MODEL_FILENAME, f"**/{_MODEL_FILENAME}"):
            hits = sorted(glob.glob(os.path.join(d, pat), recursive=True))
            if hits:
                return hits[0]
    return None


def availability_detail() -> Tuple[bool, str]:
    """返回 (是否可用, 说明)。用于界面给出精确原因，而不是笼统的"未安装"。"""
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"缺少 onnxruntime（{type(exc).__name__}）"
    path = _find_model()
    if not path:
        return False, "未找到本地模型 u2netp.onnx"
    return True, f"就绪（模型 {os.path.basename(path)}）"


def is_available() -> bool:
    return availability_detail()[0]


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
_session = None            # 复用的 onnxruntime Session（懒加载单例）
_session_model_path = None


def _get_session():
    """懒加载并复用 onnxruntime Session。"""
    global _session, _session_model_path
    if _session is not None:
        return _session

    path = _find_model()
    if not path:
        raise AiSegmentError("未找到本地 AI 模型文件 u2netp.onnx，无法离线推理")
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        raise AiSegmentError(f"onnxruntime 不可用：{exc}") from exc

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0           # 交给 onnxruntime 自动选择
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        _session_model_path = path
    except Exception as exc:  # noqa: BLE001
        raise AiSegmentError(f"AI 模型加载失败：{exc}") from exc
    return _session


def _preprocess(rgb: np.ndarray, size=_INPUT_SIZE) -> np.ndarray:
    """按 rembg 规则预处理：缩放到 320×320、归一化到 [0,1]、逐通道标准化、CHW。

    注意：中间量用 float64（与 rembg 的 numpy 默认精度一致），最后才降到
    float32。这样逐像素结果与 rembg u2netp 完全一致（实测 max diff = 0）。
    """
    from PIL import Image

    im = Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float64) / max(float(np.asarray(im).max()), 1e-6)
    out = np.empty((size[1], size[0], 3), dtype=np.float64)  # (H, W, 3)
    for c in range(3):
        out[:, :, c] = (arr[:, :, c] - _MEAN[c]) / _STD[c]
    return out.transpose(2, 0, 1)[None].astype(np.float32)   # (1, 3, 320, 320)


def predict_alpha(rgb: np.ndarray) -> np.ndarray:
    """返回与原图同尺寸的 alpha（uint8, H×W）。"""
    from PIL import Image

    sess = _get_session()
    iname = sess.get_inputs()[0].name
    try:
        outs = sess.run(None, {iname: _preprocess(rgb)})
    except Exception as exc:  # noqa: BLE001
        raise AiSegmentError(f"AI 推理失败：{exc}") from exc

    pred = np.asarray(outs[0])
    if pred.ndim == 4:
        pred = pred[:, 0, :, :]
    pred = np.squeeze(pred)
    mi, ma = float(pred.min()), float(pred.max())
    pred = (pred - mi) / (ma - mi) if ma > mi else np.zeros_like(pred)

    h, w = rgb.shape[:2]
    mask = Image.fromarray((pred * 255.0).astype(np.uint8), mode="L")
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.Resampling.LANCZOS)
    return np.asarray(mask, dtype=np.uint8)


def remove_background(rgb: np.ndarray) -> AiResult:
    """对 RGB 图像做语义分割，返回前景。

    入参 rgb: uint8, (H, W, 3)。返回 alpha 与原图同尺寸。
    """
    if not is_available():
        _, why = availability_detail()
        raise AiSegmentError(f"AI 分割不可用：{why}")

    from . import refine

    alpha = predict_alpha(rgb)
    # 前景 RGB：把边缘色向外扩，避免透明区黑边在后续合成/缩放时渗出
    rgb_out = refine.extend_edge_color(rgb, alpha)
    return AiResult(rgb=rgb_out, alpha=alpha, model_name="u2netp")


# --------------------------------------------------------------------------- #
# 混合引擎：用 AI 主体掩码补齐颜色键控误删的白色衣物
# --------------------------------------------------------------------------- #
def _keep_large_components(mask: np.ndarray, min_area_ratio: float = 0.0015,
                           min_area: int = 64) -> np.ndarray:
    """去掉 AI 掩码里面积过小的孤立块（多为背景误报），保留主体连通域。"""
    import cv2

    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    thr = max(min_area, int(mask.size * min_area_ratio))
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            out[labels == i] = 1
    return out.astype(bool)


def rescue_subject(
    color_alpha: np.ndarray,
    ai_alpha: np.ndarray,
    hi: int = 170,
    lo: int = 128,
    min_area_ratio: float = 0.0015,
) -> Tuple[np.ndarray, float]:
    """颜色键控 + AI 补齐。

    颜色键控的轮廓锐利、贴合原始边缘，但对"与背景同色的主体部件"（如白底上的
    白色衣服/头饰/头纱）无能为力——这些部件与背景色差接近，会被整块抠掉。
    AI 语义分割能识别"这是主体的一部分"，但轮廓糊、偶尔漏主体。

    本函数取长补短：**保留颜色键控的轮廓**，仅把"AI 高置信度判为主体、却被颜色
    键控判为背景"的像素补回前景。这样白衣服能保住，同时避免 AI 自身漏分割 /
    背景误报带来的新问题（AI 判背景的地方仍听颜色键控的）。

    返回 (修正后的 alpha, 被补齐的像素占比)。
    """
    import cv2

    subj = _keep_large_components(ai_alpha >= hi, min_area_ratio)
    restore = subj & (color_alpha < lo)
    if not restore.any():
        return color_alpha, 0.0

    out = color_alpha.copy()
    # 整块补齐区直接拉满到 255（硬前景），**不要**保留 AI 的软 alpha。
    #
    # 原因（2026-10-11 修复"处理后画质下降"）：AI 的 alpha 是**置信度**而非真正的
    # 透明度。restore 已限定 ai_alpha>=hi(170)，置信度足够高，本就应是前景；
    # 若把这些像素写成软值 170~249，会在主体内部留下半透明斑（实测 deepseek.jpg
    # 底部白色围裙有 4000+ 像素 alpha≈200-249；眼睛高光也被软值糊掉），
    # 叠到底板上就是发灰/透底的脏块——这正是用户感知到的"画质下降"。
    #
    # 抗锯齿交给 restore **外侧**的 AI 软 fringe（ai_alpha∈[lo,hi) 那一圈，不在 restore
    # 内，保留原 color_alpha 软值）与随后的 refine.feather 处理。这样主体内部零半透明，
    # 边界仍平滑。
    out[restore] = 255

    # 二次修洞（2026-10-11 补）：把 restore 填成 255 后，AI 主体 subj 比颜色前景大，
    # 可能把某些原本连通到图像边框的背景"围"成内部洞，其 alpha 落在 [156,249]，
    # 形成主体内部的半透明脏斑（实测 deepseek.jpg 因此多出 3423 个内部半透明像素）。
    # 再跑一次 keep_border_connected(thr=250)：只把"不与边框连通"的内部洞恢复 255，
    # 与边框连通的 genuine 背景保持原样。这样主体变实、内部零半透明。
    from . import colorkey
    out = colorkey.keep_border_connected(out)
    return out, float(restore.mean())
