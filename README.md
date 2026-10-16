# LucentCut · 一键智能去背景

<div align="center">
  <img src="assets/logo/lucentcut_banner.png" alt="LucentCut" width="720">
</div>

> 本地离线、零云服务、隐私安全的一键图片去背景工具。  
> 纯色底 / 渐变底 / 实景照片 / 3D 渲染图都能抠，发丝与细小道具清晰保留。

---

## ✨ 功能特性

- **四种抠图引擎，按需切换**
  - **精细抠图（matting，默认）**：BiRefNet / ISNet 大模型（1024×1022 级），复杂背景也能抠出发丝与飘带；纯色底自动落回颜色键控、模型缺失自动回退，旧引擎行为不受影响。
  - **混合（hybrid）**：颜色轮廓 + AI 补齐，白色衣服 / 头饰等同色部件不会丢。
  - **颜色键控（color）**：白底 / 黑底 / 纯色底最快路径。
  - **AI 语义（ai）**：u2netp 离线推理，适合复杂背景。
- **桌面 GUI（PySide6）**：打开图片 → 选引擎 → 一键处理 → 预览底板切换（红底最易看残留）→ 保存。
- **交互式裁决层**：在结果图上「保留 / 去除」涂抹、矩形框 / 描边 / 套索圈选；**颜色即语义**（绿=保留、红=去除），后画的覆盖先画的。
- **撤销 / 重写**：`Ctrl+Z` 撤销，`Ctrl+Y` 重写找回（画布笔迹与已提交修改都支持）。
- **ROI 局部处理**：框选一块区域，只重跑这一块，框外结果保持不动——适合"某处没去干净"的补救。
- **像素级细描**：画布基于 QGraphicsView（视口裁剪，高倍不卡）；描边顶点固定屏幕小点、点选笔刷可细到 1 像素，放大到看清单个像素也不挡视线。
- **命令行 / 批量**：`python -m src.main` 支持单张与整目录批量。
- **完全离线**：AI 模型随包内置（也可脚本下载），不依赖 rembg / 云服务。

## 📊 效果对比

| 引擎 | 与「其他软件理想结果」IoU | 单张耗时 |
| --- | --- | --- |
| u2netp（旧 ai） | 0.657 | 0.3–1.1s |
| **精细 matting（新）** | **0.988** | 8–17s |

> 示例：动漫多人合影、3D 角色渲染图，matting 在头发丝、手臂、细长道具上明显优于轻量模型。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows；Linux/Mac 用 source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载 AI 模型（约 400MB，断点续传）

```bash
python tools/download_models.py
```

> 模型权重**不纳入版本库**（体积过大），首次使用需执行上述命令。也可手动把模型放到 `src/models/models/`。

### 3. 运行

```bash
# 桌面 GUI
python -m src.app

# 命令行（单张 / 批量）
python -m src.main -i image/1.jpg -o output/1_nobg.png --engine matting
python -m src.main -i image/ -o output/ --batch
```

---

## 🧩 引擎说明

| 引擎 | 适用场景 | 备注 |
| --- | --- | --- |
| `matting` | 复杂背景、实景照片、3D 渲染（默认） | 大模型，较慢但最精细；纯色底自动回退颜色键控 |
| `hybrid` | 绝大多数情况、要快的纯色底 | 颜色轮廓 + AI 补回同色部件 |
| `color` | 干净白底 / 黑底 / 纯色底 | 最快，但会连白底上的白衣服一起抠掉 |
| `ai` | 复杂背景（通用语义分割） | u2netp，轻量离线 |

---

## 📁 项目结构

```
lucency_widget/
├── assets/logo/          # 品牌 logo（图标 / 组合标 / 横幅）—— 见 make_logo_assets.py
├── docs/                 # 技术方案设计 / 修改文档 / 使用功能说明
├── image/                # 测试图片（源图已纳入版本库；生成物已 gitignore）
├── src/
│   ├── app.py            # GUI 入口
│   ├── main.py           # CLI 入口
│   ├── core/             # 算法核心
│   │   ├── colorkey.py   #   颜色键控（白 / 黑 / 纯色）
│   │   ├── ai_segment.py #   AI 语义分割（自包含 u2netp 推理）
│   │   ├── hq_segment.py #   精细抠图（BiRefNet / ISNet，自包含）
│   │   ├── interactive.py#   手动抠图 + 人工裁决层
│   │   ├── refine.py     #   边缘优化
│   │   ├── pipeline.py   #   流程串联（引擎分派 / 批处理）
│   │   └── io.py         #   图片读写
│   ├── ui/               # gui.py 主窗口 / canvas.py 预览画布
│   └── models/models/    # AI 模型权重（gitignore，需下载）
├── tests/                # 端到端自测（149 项）
├── tools/                # download_models.py / make_logo_assets.py / make_comparison.py
├── build.spec            # PyInstaller 打包配置（已含图标）
├── requirements.txt
└── README.md
```

---

## 🔧 测试

```bash
python tests/test_all.py
```

当前 **149 项全部通过**，覆盖四个引擎、交互裁决语义、边界优化、模型缺失回退等。

## 📦 打包为单文件 exe

```bash
pyinstaller build.spec --clean
# 产物：dist/LucentCut.exe（含图标，约 440MB，模型随包打入）
```

> `dist/`、`build/` 等构建产物已在 `.gitignore` 排除，不会进入版本库。

---

## 📄 文档

- `docs/技术方案设计.md` —— 架构、引擎选型与决策记录
- `docs/修改文档.md` —— 逐里程碑的修改流水日志
- `docs/使用功能说明.md` —— 面向终端用户的操作指南

## 📜 开源协议

本仓库暂未指定许可证；如需用于闭源 / 商用场景，请联系作者确认。

---

<div align="center">
  <img src="assets/logo/lucentcut_lockup_light.png" alt="LucentCut" width="420">
</div>
