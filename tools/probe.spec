# -*- mode: python ; coding: utf-8 -*-
# 临时用：把 tools/frozen_probe.py 打成控制台 exe，验证冻结环境里 AI 可用。
import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.dirname(SPECPATH)          # 项目根目录
block_cipher = None
datas, binaries, hiddenimports = [], [], []
_d, _b, _h = collect_all("onnxruntime")
datas += _d
binaries += _b
hiddenimports += _h
hiddenimports += ["onnxruntime.capi._pybind_state", "src.core.ai_segment",
                  "src.core.io", "src.core.pipeline", "src.core.colorkey",
                  "src.core.refine", "src.core.interactive"]

# u2netp 模型 -> <_MEIPASS>/models/models/u2netp/u2netp.onnx
model_root = os.path.join(ROOT, "src", "models")
if os.path.isdir(model_root):
    for dirpath, _, filenames in os.walk(model_root):
        for fn in filenames:
            datas.append((os.path.join(dirpath, fn),
                          os.path.relpath(dirpath, os.path.join(ROOT, "src"))))

a = Analysis(
    [os.path.join(SPECPATH, "frozen_probe.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "skimage", "pymatting", "pooch",
              "rembg", "numba", "llvmlite", "pandas", "tkinter.test"],
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name="probe", debug=False, bootloader_ignore_signals=False, strip=False,
    upx=False, runtime_tmpdir=None, console=True, disable_windowed_traceback=False,
    argv_emulation=False, target_arch=None, codesign_identity=None,
    entitlements_file=None, icon=None,
)
