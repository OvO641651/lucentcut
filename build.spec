# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置（M3：生成单个 exe）
# 用法：.venv\Scripts\pyinstaller.exe build.spec --clean
#
# 2026-10-11 修复：原配置 hiddenimports 里塞了 rembg 并把 scipy 排除，导致
# 打包后 `import rembg` 失败（rembg.bg 顶层依赖 scipy/pymatting/skimage），
# 界面弹"AI 不可用"。现改为自包含 u2netp 推理（仅需 onnxruntime），
# 打包只需收集 onnxruntime 的动态库 + 模型文件即可。

import os

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# onnxruntime：收集全部子模块、动态库（capi DLL、provider 共享库）与数据文件
_d, _b, _h = collect_all("onnxruntime")
datas += _d
binaries += _b
hiddenimports += _h
hiddenimports += ["onnxruntime.capi._pybind_state"]

# u2netp 模型：落到 <_MEIPASS>/models/models/u2netp/u2netp.onnx
model_root = os.path.join("src", "models")
if os.path.isdir(model_root):
    for dirpath, _, filenames in os.walk(model_root):
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(dirpath, "src")   # 例如 models/models/u2netp
            datas.append((src, rel))

# 应用图标：随包打入，供运行时 QApplication.setWindowIcon 读取
# （exe 文件自身的图标由下面 EXE(icon=...) 负责；两者是独立的两件事）
_icon_src = os.path.join("assets", "logo", "lucentcut.ico")
if os.path.isfile(_icon_src):
    datas.append((_icon_src, os.path.join("assets", "logo")))

a = Analysis(
    ["src/app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 这些库本项目已不再依赖（自实现 u2netp 推理后不需要 rembg/scipy/skimage），
    # 排除可显著减小体积。
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
    name="LucentCut",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # onnxruntime 的 DLL 用 UPX 压缩易损坏，关闭
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # 桌面软件不显示黑窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/logo/lucentcut.ico",   # 由 tools/make_logo_assets.py 生成（多尺寸 16~256）
)
