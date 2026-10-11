"""LucentCut 主窗口（M2 桌面 GUI + M5 AI + M7 交互式抠图 + M9 人工裁决层
+ M12 精细抠图引擎 + M13 裁决语义修正与进度显示）。

布局：
- 左侧：文件列表 / 输入输出路径 / 参数面板 / 操作按钮
  左下角：进度条 + 处理耗时（M13 新增）
- 右侧：原图预览 + 结果预览（双栏），支持底板切换与手动抠图工具

M9 关键设计（决策 D9/D10/D13；M13 修订 D13）
---------------------------------------------
- 用户在结果图上描出的「框内保留 / 框内去除」保存为**独立的裁决层**
  （keep_mask / drop_mask，原图坐标），不参与算法计算，只在算法跑完后叠加。
  因此「先自动处理 → 再描边 → 又点一次开始处理」时裁决依然有效，
  且始终作用于最新一次算法结果。
- 裁决优先于算法；保留区若原被判成背景，用**原图 RGB** 回填颜色。
- keep 与 drop 互斥：**后画的覆盖先画的**（M13 修订）——
  新去除会撤销同一位置的旧保留，新保留会撤销旧去除；
  同一笔内重叠时以保留为准（防误删）。
- 点选涂抹的颜色在**绘制时**已定（绿=保留、红=去除），
  提交/处理时按颜色直接生效，不受「作用」下拉当前值影响（M13 修正）。
- 画布上未提交的涂抹，点「开始处理」时会自动并入裁决层（M13 新增）。
- 撤销分三层：撤销草稿点 → 撤销已闭合区域 → 撤销上一次提交的修改。
"""

from __future__ import annotations

import os
import time
import traceback
from typing import List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from src.core import ai_segment, hq_segment, interactive, io, pipeline, refine
from .canvas import Canvas
from .resources import app_icon_path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 人工裁决层边界羽化像素：0=硬边，1~2=柔边（默认 1）
OVERRIDE_FEATHER = 1
# 点选笔刷默认半径（原图像素）
BRUSH_RADIUS = 12

# 算法模式 → 界面可读名称（M12 起模式里多了 matting / hybrid-* 等前缀）
MODE_LABELS = {
    "matting": "精细抠图",
    "ai": "AI 语义分割",
    "reuse": "沿用已有透明通道",
    "hybrid": "混合引擎",
    "hybrid-white": "混合（白底）",
    "hybrid-black": "混合（黑底）",
    "hybrid-solid": "混合（纯色底）",
    "white": "颜色键控（白底）",
    "black": "颜色键控（黑底）",
    "solid": "颜色键控（纯色底）",
    "skipped": "跳过",
}


def mode_label(mode: str) -> str:
    """把内部模式名翻译成界面用语，未知模式原样返回。"""
    return MODE_LABELS.get(mode, mode)


class Worker(QtCore.QThread):
    """后台处理线程，避免大图卡 UI。"""

    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int, str)      # (当前, 总数, 文件名) 批量进度用

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LucentCut — 图片背景去除")
        # 显式设置窗口/任务栏图标（Qt 不自动继承 exe 内嵌图标）
        _icon = QtGui.QIcon(app_icon_path())
        if not _icon.isNull():
            self.setWindowIcon(_icon)
        self.resize(1280, 800)

        self._rgb: Optional[np.ndarray] = None            # 当前原图 RGB
        self._origin_alpha: Optional[np.ndarray] = None   # 输入自带的 alpha（可能为 None）
        self._alpha: Optional[np.ndarray] = None          # 当前结果 alpha
        self._out_rgb: Optional[np.ndarray] = None        # 当前结果 RGB（边缘色已外扩）
        self._src_path: Optional[str] = None
        self._worker: Optional[Worker] = None

        # 人工裁决层（原图尺寸的 uint8 掩码，255=该区域被用户标记）
        self._keep_mask: Optional[np.ndarray] = None
        self._drop_mask: Optional[np.ndarray] = None
        # 撤销历史：(keep, drop, alpha, out_rgb)
        self._history: List[Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                  Optional[np.ndarray], Optional[np.ndarray]]] = []

        # 处理计时（M13：左下角进度条 + 耗时）
        self._t0: Optional[float] = None
        self._tick_timer = QtCore.QTimer(self)
        self._tick_timer.setInterval(100)
        self._tick_timer.timeout.connect(self._on_tick)
        # 处理进行中又提交了新选区 → 完成后需重新叠加一次
        self._override_dirty: bool = False

        self._build_ui()
        self._connect()
        self._install_shortcuts()

    # ---------- UI 构建 ----------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 左侧面板
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)

        # 文件区
        file_grp = QtWidgets.QGroupBox("文件")
        fg = QtWidgets.QGridLayout(file_grp)
        self.inp_edit = QtWidgets.QLineEdit()
        self.inp_edit.setPlaceholderText("输入图片路径（可拖入文件）")
        self.inp_btn = QtWidgets.QPushButton("打开…")
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText("输出路径（留空=自动）")
        self.out_btn = QtWidgets.QPushButton("另存为…")
        fg.addWidget(self.inp_edit, 0, 0, 1, 2)
        fg.addWidget(self.inp_btn, 0, 2)
        fg.addWidget(self.out_edit, 1, 0, 1, 2)
        fg.addWidget(self.out_btn, 1, 2)
        left.addWidget(file_grp)

        # 模式区
        mode_grp = QtWidgets.QGroupBox("处理方式")
        mg = QtWidgets.QGridLayout(mode_grp)
        mg.addWidget(QtWidgets.QLabel("引擎："), 0, 0)
        self.engine_combo = QtWidgets.QComboBox()
        # 顺序即"默认"，第一项为默认引擎：精细抠图（复杂背景质量最好）
        self.engine_combo.addItem("精细抠图（推荐：复杂背景/头发最准，较慢）", "matting")
        self.engine_combo.addItem("混合（颜色轮廓 + AI 补齐，快）", "hybrid")
        self.engine_combo.addItem("颜色键控（白底/黑底，最快）", "color")
        self.engine_combo.addItem("AI 语义分割（u2netp，快但轮廓粗）", "ai")
        self.engine_combo.setToolTip(
            "精细抠图：复杂背景用 BiRefNet 大模型（1024×1024），头发丝/飘带/细长\n"
            "道具都能留住，单张约 8~9 秒；纯色背景会自动改用颜色键控轮廓（快）。\n"
            "模型缺失时自动回退到 u2netp / 颜色键控，不会失败。")
        mg.addWidget(self.engine_combo, 0, 1, 1, 2)

        mg.addWidget(QtWidgets.QLabel("背景模式："), 1, 0)
        self.mode_combo = QtWidgets.QComboBox()
        for label, val in [("自动识别", "auto"), ("白底", "white"), ("黑底", "black"),
                           ("其它纯色", "solid")]:
            self.mode_combo.addItem(label, val)
        mg.addWidget(self.mode_combo, 1, 1, 1, 2)

        mg.addWidget(QtWidgets.QLabel("预览底板："), 2, 0)
        self.bg_combo = QtWidgets.QComboBox()
        for label, val in [("棋盘格", "checker"), ("白色", "white"),
                           ("黑色", "black"), ("红色", "red")]:
            self.bg_combo.addItem(label, val)
        mg.addWidget(self.bg_combo, 2, 1, 1, 2)
        left.addWidget(mode_grp)

        # 参数区
        param_grp = QtWidgets.QGroupBox("参数")
        pg = QtWidgets.QGridLayout(param_grp)
        self.t_low = self._add_spin(pg, "软阈值下界", 30, 0, 255, 0)
        self.t_high = self._add_spin(pg, "软阈值上界", 80, 0, 255, 1)
        self.feather = self._add_spin(pg, "羽化半径(0=自动)", 0, 0, 20, 2)
        self.erode = self._add_spin(pg, "收缩像素", 0, 0, 20, 3)
        self.denoise = self._add_spin(pg, "去噪最小面积", 0, 0, 10000, 4)
        self.ov_feather = self._add_spin(pg, "手动区边缘柔化", OVERRIDE_FEATHER, 0, 5, 5)
        self.brush = self._add_spin(pg, "点选笔刷半径(px)", BRUSH_RADIUS, 2, 80, 6)
        tip = ("使用要点：下界↑=更多像素被判为背景（去得更狠）；\n"
               "上界↑=边缘过渡更宽更柔；两者之间就是半透明过渡带。\n"
               "笔刷半径=点选时一个圆点覆盖多少原图像素。\n"
               "详见《使用功能说明》第 4 / 5 章。")
        for w in (self.t_low, self.t_high, self.feather, self.erode,
                  self.denoise, self.ov_feather, self.brush):
            w.setToolTip(tip)
        left.addWidget(param_grp)

        # 操作按钮
        self.process_btn = QtWidgets.QPushButton("开始处理")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setToolTip("重新跑算法。已描出的保留/去除区域不会被清除，"
                                    "会继续叠加在新结果上；画布上还没应用的涂抹"
                                    "也会先自动并入。")
        self.batch_btn = QtWidgets.QPushButton("批量处理目录…")
        self.save_btn = QtWidgets.QPushButton("保存结果")
        self.save_btn.setEnabled(False)
        left.addWidget(self.process_btn)
        left.addWidget(self.batch_btn)
        left.addWidget(self.save_btn)

        # 状态
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        left.addWidget(self.status_label)
        left.addStretch(1)

        # 左下角：进度条 + 处理耗时（M13 新增）
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setToolTip(
            "处理中显示进度：单张为流动指示（大模型推理无法精确到百分比），\n"
            "批量为精确的「当前 / 总数」。")
        left.addWidget(self.progress_bar)

        self.time_label = QtWidgets.QLabel("处理时间：—")
        self.time_label.setStyleSheet("color: #666;")
        left.addWidget(self.time_label)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(320)
        root.addWidget(left_widget)

        # 右侧预览区
        right = QtWidgets.QVBoxLayout()
        preview_row = QtWidgets.QHBoxLayout()

        self.src_canvas = Canvas()
        self.dst_canvas = Canvas()
        self.src_canvas.set_tool("none")
        self.dst_canvas.set_tool("none")

        src_col = QtWidgets.QVBoxLayout()
        src_col.addWidget(QtWidgets.QLabel("原图（滚轮缩放 / 中键拖动）"))
        src_col.addWidget(self.src_canvas, 1)
        dst_col = QtWidgets.QVBoxLayout()
        dst_col.addWidget(QtWidgets.QLabel("结果（可在此手动抠图）"))
        dst_col.addWidget(self.dst_canvas, 1)

        preview_row.addLayout(src_col, 1)
        preview_row.addLayout(dst_col, 1)
        right.addLayout(preview_row, 1)

        # 手动抠图工具栏：绘制方式
        draw_bar = QtWidgets.QHBoxLayout()
        draw_bar.addWidget(QtWidgets.QLabel("绘制："))
        self.tool_none = QtWidgets.QPushButton("浏览")
        self.tool_rect = QtWidgets.QPushButton("矩形框")
        self.tool_points = QtWidgets.QPushButton("点选")
        self.tool_polygon = QtWidgets.QPushButton("描边")
        self.tool_lasso = QtWidgets.QPushButton("套索")
        for b in (self.tool_none, self.tool_rect, self.tool_points,
                  self.tool_polygon, self.tool_lasso):
            b.setCheckable(True)
            draw_bar.addWidget(b)
        self.tool_none.setChecked(True)
        right.addLayout(draw_bar)

        # 手动抠图工具栏：作用 + 执行
        act_bar = QtWidgets.QHBoxLayout()
        act_bar.addWidget(QtWidgets.QLabel("作用："))
        self.action_combo = QtWidgets.QComboBox()
        self.action_combo.addItem("保留（画的地方一定留下）", "keep")
        self.action_combo.addItem("去除（画的地方一定去掉）", "drop")
        self.action_combo.setToolTip(
            "决定「作用：」下拉里选的是什么，以及「点选」左键涂的颜色。\n"
            "保留：把被算法误删的部分找回来（用原图颜色，如白裙子、眼睛里的白色）；\n"
            "去除：把没去干净的部分删掉。\n"
            "点选时：左键=当前作用，右键=反面（一笔里既能补也能删）；\n"
            "矩形框/描边/套索：圈出的区域整体按当前作用处理。\n"
            "同一处先画保留、后画去除（或反过来）时：后画的覆盖先画的。")
        act_bar.addWidget(self.action_combo, 1)

        self.apply_btn = QtWidgets.QPushButton("✔ 应用选区 (Enter)")
        self.apply_btn.setToolTip("把当前画好的所有涂抹/区域一次性生效（可一次画多个区域）")
        self.undo_btn = QtWidgets.QPushButton("↶ 撤销 (Ctrl+Z)")
        self.clear_sel_btn = QtWidgets.QPushButton("清空选区 (Esc)")
        self.clear_ov_btn = QtWidgets.QPushButton("清除手动修改")
        self.fit_btn = QtWidgets.QPushButton("适应窗口 (Ctrl+0)")
        for b in (self.apply_btn, self.undo_btn, self.clear_sel_btn,
                  self.clear_ov_btn, self.fit_btn):
            act_bar.addWidget(b)
        right.addLayout(act_bar)

        # 选项行：显示修改区 / GrabCut 自动抠图
        opt_bar = QtWidgets.QHBoxLayout()
        self.show_ov = QtWidgets.QCheckBox("显示修改区（绿=保留 红=去除）")
        self.show_ov.setChecked(False)
        self.show_ov.setToolTip(
            "默认**关闭**：应用/处理完成后只显示结果，不再残留绿色描边。\n"
            "勾上可以查看自己标过哪些区域。")
        opt_bar.addWidget(self.show_ov)

        self.grabcut_btn = QtWidgets.QPushButton("GrabCut 自动抠图")
        self.grabcut_btn.setToolTip(
            "用「矩形框 / 描边 / 套索」圈出一个大致范围，让 GrabCut 自动分离主体。\n"
            "会**重新计算整张图的 alpha**（不同于「保留/去除」只改局部）。\n"
            "画的区域越大越准；至少要画一个闭合区域。")
        opt_bar.addWidget(self.grabcut_btn)
        opt_bar.addStretch(1)
        right.addLayout(opt_bar)

        # 初始化画布的笔刷与作用色（此时控件都已创建）
        self.dst_canvas.set_brush_radius(int(self.brush.value()))
        self.dst_canvas.set_paint_mode(self.action_combo.currentData())

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 1)

    def _add_spin(self, grid, label, default, lo, hi, row):
        grid.addWidget(QtWidgets.QLabel(label), row, 0)
        sp = QtWidgets.QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(default)
        grid.addWidget(sp, row, 1)
        return sp

    # ---------- 信号连接 ----------
    def _connect(self):
        self.inp_btn.clicked.connect(self._open_file)
        self.out_btn.clicked.connect(self._save_file)
        self.process_btn.clicked.connect(self._process)
        self.batch_btn.clicked.connect(self._batch)
        self.save_btn.clicked.connect(self._save_result)
        self.bg_combo.currentIndexChanged.connect(self._on_bg_changed)

        self.tool_none.clicked.connect(lambda: self._set_tool("none"))
        self.tool_rect.clicked.connect(lambda: self._set_tool("rect"))
        self.tool_points.clicked.connect(lambda: self._set_tool("fg"))
        self.tool_polygon.clicked.connect(lambda: self._set_tool("polygon"))
        self.tool_lasso.clicked.connect(lambda: self._set_tool("lasso"))

        self.apply_btn.clicked.connect(self._commit_selection)
        self.undo_btn.clicked.connect(self._undo)
        self.clear_sel_btn.clicked.connect(self._clear_selection)
        self.clear_ov_btn.clicked.connect(self._clear_override)
        self.fit_btn.clicked.connect(self._fit_all)
        self.grabcut_btn.clicked.connect(self._grabcut)
        self.show_ov.toggled.connect(self._on_show_override)
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        self.brush.valueChanged.connect(self._on_brush_changed)

        self.dst_canvas.selectionCommitted.connect(self._on_commit)
        self.dst_canvas.selectionChanged.connect(self._on_selection_changed)
        self.dst_canvas.undoRequested.connect(self._undo)

        self.setAcceptDrops(True)

    def _install_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence.Undo, self, self._undo)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self,
                        self._commit_selection)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self,
                        self._commit_selection)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self, self._fit_all)

    # ---------- 文件操作 ----------
    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            self._load(path)

    def _save_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存输出", self._default_out(), "PNG (*.png)")
        if path:
            self.out_edit.setText(path)

    def _default_out(self) -> str:
        if self._src_path:
            return os.path.splitext(self._src_path)[0] + "_nobg.png"
        return "output.png"

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if os.path.splitext(p)[1].lower() in IMG_EXT:
                self._load(p)
                break

    def _load(self, path: str):
        try:
            self._rgb, self._origin_alpha = io.load_image(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "加载失败", str(exc))
            return
        self._src_path = path
        self.inp_edit.setText(path)
        self._alpha = None
        self._out_rgb = None
        # 换图即清空人工裁决层与历史
        self._keep_mask = None
        self._drop_mask = None
        self._history = []
        self.src_canvas.set_image(self._rgb, None, self._bg(), keep_transform=False)
        self.src_canvas.set_override_masks(None, None)
        self.dst_canvas.set_image(self._rgb, None, self._bg(), keep_transform=False)
        self._refresh_overlay()
        self.dst_canvas.clear_selection()
        self.dst_canvas.set_tool("none")
        self._set_tool("none")
        self.save_btn.setEnabled(False)
        extra = "，含透明通道" if self._origin_alpha is not None else ""
        self.status_label.setText(
            f"已加载：{os.path.basename(path)}"
            f"（{self._rgb.shape[1]}×{self._rgb.shape[0]}{extra}）— 点「开始处理」")

    # ---------- 处理 ----------
    def _kwargs(self) -> dict:
        return dict(
            mode=self.mode_combo.currentData(),
            t_low=float(self.t_low.value()),
            t_high=float(self.t_high.value()),
            feather_radius=self.feather.value() or None,   # 0 = 自动（按尺寸）
            erode_px=self.erode.value(),
            denoise_min_area=self.denoise.value(),
            engine=self.engine_combo.currentData(),
            keep_mask=self._keep_mask,
            drop_mask=self._drop_mask,
            override_feather=int(self.ov_feather.value()),
        )

    def _process(self):
        if self._rgb is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先打开一张图片")
            return
        # 防重入：上一个处理还没结束时忽略（按钮虽已禁用，但 Enter 快捷键
        # 与「应用选区」的自动处理仍可能触发；并发跑两个大模型会内存爆）
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("正在处理中，请稍候…")
            return
        # M13：画布上还有未应用的涂抹/区域时，先并入裁决层再处理。
        # 修复"画完点选直接点『开始处理』，点选不生效、圆点残留在画面上"。
        if self.dst_canvas.has_pending():
            regions, fg_pts, bg_pts = self.dst_canvas.take_selection()
            keep_new, drop_new = self._build_selection_masks(regions, fg_pts, bg_pts)
            if keep_new.any() or drop_new.any():
                self._push_history()
                self._keep_mask, self._drop_mask = interactive.merge_override_masks(
                    self._keep_mask, self._drop_mask, keep_new, drop_new)
        engine = self.engine_combo.currentData()
        if engine == "ai":
            ok, why = ai_segment.availability_detail()
            if not ok:
                QtWidgets.QMessageBox.warning(
                    self, "AI 不可用",
                    f"AI 语义分割当前不可用：\n{why}\n\n"
                    "AI 模型（u2netp.onnx）已随程序附带，"
                    "通常无需额外安装；若仍不可用，请用「颜色键控」引擎。")
                return
        if engine == "matting":
            # 精细模型缺失不算错误：pipeline 会自动回退到 u2netp/颜色键控，
            # 这里只提前告知一声，避免用户以为"选了精细却没变精细"。
            ok, why = hq_segment.availability_detail()
            if not ok:
                QtWidgets.QMessageBox.information(
                    self, "精细模型不可用",
                    f"精细抠图模型当前不可用：\n{why}\n\n"
                    "本次将自动回退到 u2netp / 颜色键控（结果仍可用，"
                    "但复杂背景的头发与细轮廓会差一些）。\n\n"
                    "如需最好效果：运行 tools/download_models.py 下载 BiRefNet。")
        self._set_busy(True, "处理中…")
        self._override_dirty = False
        self._worker = Worker(self._run_process, self._rgb,
                              self._origin_alpha, self._kwargs())
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _run_process(self, rgb, orig_alpha, kwargs) -> dict:
        """在线程里执行，返回结果 dict。逻辑与 CLI/pipeline 完全一致。"""
        kw = dict(kwargs)
        # 人工裁决层不能带进批量/纯算法路径之外的比较，这里保留即可
        rgb_out, alpha, mode, rescued, reused, warnings = pipeline.process_array(
            rgb, orig_alpha, **kw)
        return {"rgb": rgb_out, "alpha": alpha, "mode": mode,
                "rescued": rescued, "reused": reused, "warnings": warnings}

    def _on_done(self, result: dict):
        dt = self._set_busy(False)
        self._alpha = result["alpha"]
        self._out_rgb = result["rgb"]
        self.dst_canvas.set_image(result["rgb"], result["alpha"], self._bg())
        self._refresh_overlay()
        self.save_btn.setEnabled(True)
        msg = f"完成（{mode_label(result['mode'])}，耗时 {dt:.1f}s）"
        if result.get("rescued"):
            msg += f" — AI 补齐 {result['rescued']:.1%}"
        if self._has_override():
            msg += " — 已叠加手动修改"
        if result["warnings"]:
            msg += " — " + "；".join(result["warnings"])
        self.status_label.setText(msg)
        # 处理期间又提交了新选区 → 用最新裁决层重新叠加一次
        if self._override_dirty and self._has_override():
            self._override_dirty = False
            self._apply_override_local()

    def _on_failed(self, err: str):
        self._set_busy(False)
        self.status_label.setText("失败：" + err)
        QtWidgets.QMessageBox.critical(self, "处理失败", err)

    def _set_busy(self, busy: bool, text: str = "", total: int = 0) -> float:
        """进入/退出忙碌态，驱动左下角进度条与耗时显示。

        total > 0：批量模式，进度条精确到「当前/总数」；
        否则单张/GrabCut：进度条为流动指示（推理无法精确到百分比）。
        返回退出忙碌态时的耗时（秒）。
        """
        self.process_btn.setEnabled(not busy)
        self.batch_btn.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy)
        if busy:
            self._t0 = time.perf_counter()
            self._tick_timer.start()
            if total > 0:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(0)
            else:
                self.progress_bar.setRange(0, 0)      # 忙指示（来回流动）
            if text:
                self.status_label.setText(text)
            return 0.0
        self._tick_timer.stop()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        dt = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        self._t0 = None
        self.time_label.setText(f"上次处理耗时：{dt:.1f}s")
        return dt

    def _on_tick(self):
        """处理中每 100ms 刷新一次耗时显示。"""
        if self._t0 is not None:
            self.time_label.setText(
                f"处理中… {time.perf_counter() - self._t0:.1f}s")

    def _save_result(self):
        if self._alpha is None:
            return
        out = self.out_edit.text().strip() or self._default_out()
        rgb = self._out_rgb if self._out_rgb is not None else self._rgb
        try:
            io.save_rgba(rgb, self._alpha, out)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.status_label.setText(f"已保存：{out}")

    # ---------- 批量 ----------
    def _batch(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输入目录")
        if not d:
            return
        out_dir = os.path.join(d, "output")
        kw = self._kwargs()
        # 批量处理不套用当前图的手动裁决层（那是针对单张的）
        kw.pop("keep_mask", None)
        kw.pop("drop_mask", None)
        # 统计图片总数，进度条按「当前/总数」精确显示（M13）
        total = sum(
            1 for name in os.listdir(d)
            if os.path.isfile(os.path.join(d, name))
            and os.path.splitext(name)[1].lower() in IMG_EXT)
        # 进度回调在工作线程里执行，经 Worker.progress 信号转发到主线程
        cell: dict = {}

        def _cb(cur: int, tot: int, name: str):
            w = cell.get("w")
            if w is not None:
                w.progress.emit(cur, tot, name)

        kw["progress_cb"] = _cb
        self._set_busy(True, f"批量处理中…（共 {total} 张）", total=total)
        self._worker = Worker(pipeline.process_batch, d, out_dir, **kw)
        cell["w"] = self._worker
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.done.connect(self._on_batch_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_batch_progress(self, cur: int, total: int, name: str):
        self.progress_bar.setValue(cur)
        self.status_label.setText(f"批量处理中… {cur}/{total}（{name}）")

    def _on_batch_done(self, results: List):
        dt = self._set_busy(False)
        ok = sum(1 for r in results if r.mode != "skipped")
        skipped = sum(1 for r in results if r.mode == "skipped")
        self.status_label.setText(
            f"批量完成：成功 {ok} 张，跳过 {skipped} 张，"
            f"耗时 {dt:.1f}s，输出到 output/")

    # ---------- 视图 ----------
    def _bg(self) -> str:
        return self.bg_combo.currentData()

    def _on_bg_changed(self):
        bg = self._bg()
        if self._rgb is not None:
            self.src_canvas.set_image(self._rgb, None, bg)
        if self._alpha is not None:
            rgb = self._out_rgb if self._out_rgb is not None else self._rgb
            self.dst_canvas.set_image(rgb, self._alpha, bg)

    def _fit_all(self):
        self.src_canvas.fit_view()
        self.dst_canvas.fit_view()

    # ---------- 手动抠图 / 人工裁决 ----------
    def _set_tool(self, tool: str):
        mapping = {"none": self.tool_none, "rect": self.tool_rect,
                   "fg": self.tool_points, "polygon": self.tool_polygon,
                   "lasso": self.tool_lasso}
        for t, btn in mapping.items():
            btn.setChecked(t == tool)
        self.dst_canvas.set_tool(tool)
        if tool != "none":
            self.dst_canvas.setFocus()
            self.status_label.setText(self._tool_hint(tool))

    def _tool_hint(self, tool: str) -> str:
        action = self.action_combo.currentData()
        what = {"keep": "保留", "drop": "去除"}.get(action, "保留")
        base = {
            "rect": "拖出一个矩形框",
            "fg": f"按住左键涂「{what}」，右键涂反面（笔刷半径见左侧参数）",
            "polygon": "沿轮廓逐个点顶点，双击或右键闭合；可继续画下一个区域",
            "lasso": "按住左键沿轮廓画一圈，松开闭合；可继续画下一个区域",
        }.get(tool, "")
        return (f"{base}；作用={what} → 画完按 Enter 或点「应用选区」生效"
                "（Esc 清空，Ctrl+Z 撤销）")

    def _clear_selection(self):
        self.dst_canvas.clear_selection()
        self.status_label.setText("已清空未提交的选区（已生效的修改不受影响）")

    def _commit_selection(self):
        if self.dst_canvas.tool() == "none":
            self.status_label.setText("请先选择一个绘制工具（矩形框/点选/描边/套索）")
            return
        if not self.dst_canvas.commit():
            self.status_label.setText("当前没有可应用的选区")

    def _on_selection_changed(self, n_regions: int, n_fg: int, n_bg: int):
        total = n_regions + n_fg + n_bg
        self.apply_btn.setEnabled(total > 0)
        if total:
            self.status_label.setText(
                f"已画 {n_regions} 个区域 / {n_fg} 个保留笔刷点 / {n_bg} 个去除笔刷点"
                " — 按 Enter 或点「应用选区」生效")

    def _on_commit(self, regions, fg_pts, bg_pts):
        """应用选区：把涂抹点/区域写进裁决层并立刻生效。

        这里是**直接裁决**，不做 GrabCut、也不重跑算法：
        - 点选（fg_pts / bg_pts）→ 圆盘掩码，半径 = 笔刷半径。
          **颜色语义在绘制时已定**：fg_pts=保留（绿）、bg_pts=去除（红），
          提交时按颜色直接生效，**不受「作用」下拉当前值影响**
          （M13 修正：修复"点选去除被当成保留"的缺陷——旧实现把 bg_pts
          一律当作当前作用的反面，选『去除』时左键涂的去除点被误判为保留）；
        - 矩形框 / 描边 / 套索（regions）无颜色语义，按「作用」下拉的
          当前值整体处理；
        - **后画的覆盖先画的**（M13，决策 D13 修订）：新去除撤销旧保留、
          新保留撤销旧去除（旧实现无条件保留优先，导致"先保留过的区域，
          后画的去除永远不生效"）。
        需要「让算法重新抠一遍」请用下方独立的「GrabCut 自动抠图」按钮。
        """
        if self._rgb is None:
            return
        if not regions and not fg_pts and not bg_pts:
            self.status_label.setText("还没有画出任何涂抹/区域")
            return

        keep_new, drop_new = self._build_selection_masks(regions, fg_pts, bg_pts)
        if not keep_new.any() and not drop_new.any():
            self.status_label.setText("选区太小，没有产生任何像素；可调大笔刷半径再试")
            return

        self._push_history()
        self._keep_mask, self._drop_mask = interactive.merge_override_masks(
            self._keep_mask, self._drop_mask, keep_new, drop_new)
        # 处理进行中时：选区已并入裁决层，等当前处理完成后自动重新叠加
        # （在跑的那次处理用的是启动时的掩码快照，不含这笔新画的）
        if self._worker is not None and self._worker.isRunning():
            self._override_dirty = True
            self.dst_canvas.clear_selection()
            self.status_label.setText("处理中：选区已记录，本次处理完成后自动生效")
            return
        self._apply_override_local()
        self.dst_canvas.clear_selection()
        self.dst_canvas.setFocus()

    def _build_selection_masks(self, regions, fg_pts, bg_pts):
        """把画布提交的选区栅格化成 (keep_new, drop_new) 两个新掩码。

        点选的颜色在绘制时已定：fg_pts=保留（绿）、bg_pts=去除（红）。
        区域（矩形框/描边/套索）无颜色语义，按「作用」下拉当前值整体处理。
        """
        shape = self._rgb.shape
        brush = int(self.brush.value())
        keep_new = np.zeros(shape[:2], np.uint8)
        drop_new = np.zeros(shape[:2], np.uint8)

        if regions:
            action = self.action_combo.currentData()
            m = interactive.regions_to_mask(shape, regions)
            (keep_new if action == "keep" else drop_new)[m > 0] = 255
        if fg_pts:                               # 绿点 = 保留
            m = interactive.points_to_mask(shape, fg_pts, brush)
            keep_new[m > 0] = 255
        if bg_pts:                               # 红点 = 去除
            m = interactive.points_to_mask(shape, bg_pts, brush)
            drop_new[m > 0] = 255
        return keep_new, drop_new

    def _grabcut(self):
        """用画好的区域作为先验，跑一次 GrabCut 重新计算整张图的 alpha。"""
        if self._rgb is None:
            self.status_label.setText("请先打开一张图片")
            return
        regions = self.dst_canvas.regions()
        if not regions:
            self.status_label.setText(
                "请先用「矩形框 / 描边 / 套索」圈出主体的大致范围，再点 GrabCut 自动抠图")
            return
        self._push_history()
        self._set_busy(True, "GrabCut 抠图中…")
        self._worker = Worker(self._run_matte, regions)
        self._worker.done.connect(self._on_interactive_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()
        self.dst_canvas.clear_selection()

    def _run_matte(self, regions) -> dict:
        rgb_out, alpha = interactive.matting_from_regions(self._rgb, regions)
        alpha = refine.harden_alpha(alpha)
        if self._origin_alpha is not None:
            alpha = refine.merge_alpha(alpha, self._origin_alpha)
        rgb_out, alpha = interactive.apply_override(
            rgb_out, alpha, self._rgb, self._keep_mask, self._drop_mask,
            feather_px=int(self.ov_feather.value()))
        return {"rgb": rgb_out, "alpha": alpha, "mode": "manual-matte",
                "warnings": [f"按 {len(regions)} 个区域 GrabCut 自动抠图"]}

    def _on_interactive_done(self, result: dict):
        dt = self._set_busy(False)
        self._alpha = result["alpha"]
        self._out_rgb = result["rgb"]
        self.dst_canvas.set_image(result["rgb"], result["alpha"], self._bg())
        self._refresh_overlay()
        self.dst_canvas.clear_selection()
        self.save_btn.setEnabled(True)
        msg = f"GrabCut 抠图完成（耗时 {dt:.1f}s）"
        if self._has_override():
            msg += "（已叠加保留/去除裁决）"
        self.status_label.setText(msg + "，可继续调整或保存")

    # ---------- 裁决层工具 ----------
    def _has_override(self) -> bool:
        return (self._keep_mask is not None and self._keep_mask.any()) or \
               (self._drop_mask is not None and self._drop_mask.any())

    def _apply_override_local(self):
        """不重跑算法，只把裁决层叠加到当前结果上。"""
        if self._rgb is None:
            return
        if self._alpha is None:
            # 还没处理过 → 先跑一遍算法（process 内部会带上裁决层）
            self._process()
            return
        rgb = self._out_rgb if self._out_rgb is not None else self._rgb
        rgb_out, alpha = interactive.apply_override(
            rgb, self._alpha, self._rgb, self._keep_mask, self._drop_mask,
            feather_px=int(self.ov_feather.value()))
        self._alpha = alpha
        self._out_rgb = rgb_out
        self.dst_canvas.set_image(rgb_out, alpha, self._bg())
        self._refresh_overlay()
        self.save_btn.setEnabled(True)
        n_k = int((self._keep_mask > 0).sum()) if self._keep_mask is not None else 0
        n_d = int((self._drop_mask > 0).sum()) if self._drop_mask is not None else 0
        self.status_label.setText(
            f"已生效：保留 {n_k} px、去除 {n_d} px"
            "（结果已更新；勾选「显示修改区」可回看标注，Ctrl+Z 可撤销）")

    def _refresh_overlay(self):
        """按「显示修改区」开关决定是否在画布上显示绿/红标注。

        默认**关闭**：应用/处理完成后只保留结果画面，
        不再残留绿色描边（用户反馈的问题 ①）。
        """
        if getattr(self, "show_ov", None) is not None and self.show_ov.isChecked():
            self.dst_canvas.set_override_masks(self._keep_mask, self._drop_mask)
        else:
            self.dst_canvas.set_override_masks(None, None)

    def _on_show_override(self, _checked: bool = False):
        self._refresh_overlay()

    def _on_action_changed(self, _idx: int = 0):
        self.dst_canvas.set_paint_mode(self.action_combo.currentData())
        t = self.dst_canvas.tool()
        if t != "none":
            self.status_label.setText(self._tool_hint(t))

    def _on_brush_changed(self, v: int):
        self.dst_canvas.set_brush_radius(int(v))
        if self.dst_canvas.tool() == "fg":
            self.status_label.setText(f"笔刷半径已设为 {int(v)} 像素")

    def _push_history(self):
        """提交前保存快照，供撤销回退。大图限制深度，避免吃内存。"""
        if self._rgb is None:
            return
        area = self._rgb.shape[0] * self._rgb.shape[1]
        limit = 3 if area > 8_000_000 else (6 if area > 2_000_000 else 12)
        self._history.append((
            None if self._keep_mask is None else self._keep_mask.copy(),
            None if self._drop_mask is None else self._drop_mask.copy(),
            None if self._alpha is None else self._alpha.copy(),
            None if self._out_rgb is None else self._out_rgb.copy(),
        ))
        if len(self._history) > limit:
            self._history.pop(0)

    def _undo(self):
        """撤销：先退画布上的草稿/区域，再退上一次已提交的修改。

        注意：这里必须用 has_pending() 判断而不是直接调 canvas.undo()。
        后者在画布为空时会发 undoRequested 回来，而本函数又去调 canvas.undo()，
        形成「主窗口 → 画布 → 主窗口」的无限递归（实测直接把进程栈打爆）。
        """
        if self.dst_canvas.has_pending():
            self.dst_canvas.undo()
            return
        if not self._history:
            self.status_label.setText("没有可撤销的操作")
            return
        keep, drop, alpha, out_rgb = self._history.pop()
        self._keep_mask = keep
        self._drop_mask = drop
        self._alpha = alpha
        self._out_rgb = out_rgb
        if alpha is not None:
            rgb = out_rgb if out_rgb is not None else self._rgb
            self.dst_canvas.set_image(rgb, alpha, self._bg())
            self.save_btn.setEnabled(True)
        else:
            self.dst_canvas.set_image(self._rgb, None, self._bg())
            self.save_btn.setEnabled(False)
        self._refresh_overlay()
        self.status_label.setText("已撤销上一步修改")

    def _clear_override(self):
        """清除全部人工裁决，回到纯算法结果。"""
        if not self._has_override():
            self.status_label.setText("当前没有手动修改")
            return
        self._push_history()
        self._keep_mask = None
        self._drop_mask = None
        # 画布上未提交的涂抹一并清掉，避免「清除手动修改」后又把它们并回来
        self.dst_canvas.clear_selection()
        self._refresh_overlay()
        self._process()
