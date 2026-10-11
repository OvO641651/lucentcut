# -*- mode: python ; coding: utf-8 -*-
# 冻结环境探针（与 build.spec 同一套依赖收集，只把入口换成 tools/probe_gui.py 并开控制台）

import os

from PyInstaller.utils.hooks import collect_all

# spec 会被以 tools/ 为工作目录执行，统一用绝对路径避免层级错位
# SPEC = <项目根>/tools/probe_gui.spec → 上两级才是项目根
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))

datas = []
binaries = []
hiddenimports = []

_d, _b, _h = collect_all("onnxruntime")
datas += _d
binaries += _b
hiddenimports += _h
hiddenimports += ["onnxruntime.capi._pybind_state"]

model_root = os.path.join(ROOT, "src", "models")
if os.path.isdir(model_root):
    for dirpath, _, filenames in os.walk(model_root):
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(dirpath, os.path.join(ROOT, "src"))
            datas.append((src, rel))

a = Analysis(
    [os.path.join(ROOT, "tools", "probe_gui.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "skimage", "pymatting", "pooch",
        "rembg", "numba", "llvmlite", "pandas", "notebook", "IPython",
        "tkinter.test",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="probe_gui",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
