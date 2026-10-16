"""预览画布组件（M2 + M7 + M9 + M15 迁移到 QGraphicsView）。

M15 迁移说明（2026-10-16）
--------------------------
旧实现继承 QWidget，`paintEvent` 里对**整张图**执行
`self._pixmap.scaled(target_w, target_h, ...)`：`scale=8` 时一张 4000×3000
的图会先生成 32000×24000（≈7.7 亿像素、约 3GB）的中间位图，而实际只显示
窗口大小的一小块 —— 高倍缩放时每动一下鼠标都全量重算，于是卡顿。

现改为 **QGraphicsView / QGraphicsScene**：
- 内容放在一个自定义 `QGraphicsItem` 里，其 `boundingRect()` = 原图矩形；
- 缩放/平移通过该 item 的 `setScale()` / `setPos()` 表达；
- Qt 只对**暴露区域（视口）**做光栅化，**不再产生整图超大中间位图**，
  绘制量降到窗口大小（约 1280×800），根治高倍缩放卡顿。

坐标语义**完全沿用旧版**（这是刻意的）：`_scale` / `_offset` / `_to_image()` /
`_to_widget()` 保持与 QWidget 版一致。因此 `gui.py` 与全部自测无需改动，
交互、平移、撤销等行为也一字不变；本次只替换**渲染层**。

其它职责（不变）：
- 显示原图 / 结果图，支持底板切换（棋盘格 / 白 / 黑 / 红）
- 视图操作：滚轮缩放、中键（或 Ctrl+左键）拖动平移、适应窗口
- 手动抠图交互：矩形框选、点选（前景/背景）、描边多边形、套索自由曲线
- 一次可描**多个互不相连的区域**，支持逐级撤销 / **重写（redo）**
- 新增 **ROI 处理区**（工具 "roi"）：框选后只对该区域跑引擎，区外不变
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

# 区域颜色：已闭合区域用青绿，正在画的草稿用亮绿
_COLOR_REGION = QtGui.QColor(46, 204, 113)
_COLOR_DRAFT = QtGui.QColor(120, 220, 160)
_COLOR_FG = QtGui.QColor(46, 204, 113)
_COLOR_BG = QtGui.QColor(231, 76, 60)
_COLOR_KEEP = QtGui.QColor(46, 204, 113)
_COLOR_DROP = QtGui.QColor(231, 76, 60)
_COLOR_ROI = QtGui.QColor(240, 170, 40)     # ROI 处理区（琥珀色，虚线）

# 缩放上下限（旧版上限 32× 会产生超大位图，收窄到 16×，仍够"看清单个像素"）
_MIN_SCALE = 0.02
_MAX_SCALE = 16.0

_VERTEX_DOT_PX = 3.0


def vertex_dot_pen(color: QtGui.QColor) -> QtGui.QPen:
    """顶点圆点用的画笔：**cosmetic**（宽度按设备像素恒定）+ 圆头。

    用 cosmetic 画笔配合 `drawPoint` 画点，等价于"屏幕直径恒为 _VERTEX_DOT_PX"，
    且在高倍缩放下不会像亚像素椭圆那样被栅格化吃掉（16× 时仍可见）。
    """
    pen = QtGui.QPen(color, _VERTEX_DOT_PX, QtCore.Qt.SolidLine,
                     QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
    pen.setCosmetic(True)
    return pen


def make_checker_pixmap(w: int, h: int, cell: int = 12) -> QtGui.QPixmap:
    """生成棋盘格底图（透明区域的视觉提示）。"""
    img = Image.new("RGB", (max(w, 1), max(h, 1)), (255, 255, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            col = (240, 240, 240) if (x // cell + y // cell) % 2 == 0 else (205, 205, 205)
            d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=col)
    return _pil_to_pixmap(img)


def _pil_to_pixmap(img: Image.Image) -> QtGui.QPixmap:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, img.width, img.height, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def composite_to_pixmap(rgb: np.ndarray, alpha: Optional[np.ndarray],
                        bg: str = "checker") -> QtGui.QPixmap:
    """把 rgb(+alpha) 合成到底板上，返回 QPixmap。

    bg: "checker" | "white" | "black" | "red"
    """
    h, w = rgb.shape[:2]
    if alpha is None:
        alpha = np.full((h, w), 255, np.uint8)
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    fg = Image.fromarray(rgba, "RGBA")

    if bg == "white":
        base = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    elif bg == "black":
        base = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    elif bg == "red":
        base = Image.new("RGBA", (w, h), (230, 30, 40, 255))
    else:  # checker
        base = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        from PIL import ImageDraw
        d = ImageDraw.Draw(base)
        cell = 12
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                col = (240, 240, 240, 255) if (x // cell + y // cell) % 2 == 0 else (205, 205, 205, 255)
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=col)

    base.paste(fg, (0, 0), fg)
    return _pil_to_pixmap(base)


def _qimage_from_mask(mask: np.ndarray, color: QtGui.QColor, alpha: int = 96) -> QtGui.QImage:
    """把二值掩码转成半透明彩色 QImage（用于绘制人工裁决区）。"""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    m = mask > 0
    out[m, 0] = color.red()
    out[m, 1] = color.green()
    out[m, 2] = color.blue()
    out[m, 3] = alpha
    return QtGui.QImage(out.tobytes(), w, h, QtGui.QImage.Format_RGBA8888).copy()


class _ContentItem(QtWidgets.QGraphicsItem):
    """画布内容项：在**原图像素坐标**下绘制图像、裁决层与选区。

    boundingRect() 返回原图矩形 (0,0,W,H)；缩放/平移由 Canvas 通过
    setScale()/setPos() 施加到本项上，从而交给 Qt 做视口裁剪。
    """

    def __init__(self, canvas: "Canvas"):
        super().__init__()
        self._canvas = canvas
        self._w = 1
        self._h = 1

    def set_geometry(self, w: int, h: int) -> None:
        self.prepareGeometryChange()
        self._w, self._h = max(1, int(w)), max(1, int(h))

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0.0, 0.0, float(self._w), float(self._h))

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        self._canvas._paint_content(painter)


class Canvas(QtWidgets.QGraphicsView):
    """预览画布：QGraphicsView 渲染 + 多区域交互式选区绘制。"""

    # 选区变化（用于更新状态栏与按钮可用性）
    selectionChanged = QtCore.Signal(int, int, int)        # (区域数, 前景点数, 背景点数)
    # 用户按下 Enter / 点「应用选区」：一次性提交全部区域与种子点
    selectionCommitted = QtCore.Signal(list, list, list)   # (regions, fg_pts, bg_pts)
    # 画布内无可撤销内容时请求主窗口回退上一次已提交的修改
    undoRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        # 视图变换保持单位阵：缩放/平移都放在内容项上（场景坐标 = 视图像素坐标）
        self.setRenderHints(QtGui.QPainter.Antialiasing
                            | QtGui.QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.setBackgroundBrush(QtGui.QColor(35, 35, 38))
        self.setMouseTracking(True)
        self.setMinimumSize(320, 320)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._content = _ContentItem(self)
        self._scene.addItem(self._content)

        self._pixmap: Optional[QtGui.QPixmap] = None
        self._bg = "checker"
        self._tool = "none"
        self._scale = 1.0
        self._offset = QtCore.QPoint(0, 0)

        # 交互状态
        self._drawing = False              # 矩形 / 套索 / ROI 拖拽中
        self._start: Optional[Tuple[int, int]] = None
        self._cur: Optional[Tuple[int, int]] = None
        self._draft: List[Tuple[int, int]] = []        # 正在画的区域的点
        self._regions: List[List[Tuple[int, int]]] = []  # 已闭合的区域
        self._fg_pts: List[Tuple[int, int]] = []       # 点选：绿=保留
        self._bg_pts: List[Tuple[int, int]] = []       # 点选：红=去除

        # ROI 处理区（独立图层，不受 clear_selection 影响）
        self._roi_draft: List[Tuple[int, int]] = []
        self._roi_regions: List[List[Tuple[int, int]]] = []

        # 重写（redo）栈：保存"撤销前"的选区状态快照
        self._redo: List[tuple] = []

        # 点选笔刷（直接裁决，不走 GrabCut）
        self._brush_radius = 12            # 按原图像素计
        self._paint_mode = "keep"          # 左键涂什么颜色：keep / drop
        self._painting = False
        self._last_paint: Optional[Tuple[int, int]] = None

        # 视图状态（平移用 offset 位移，保持旧版语义）
        self._panning = False
        self._pan_start: Optional[QtCore.QPoint] = None
        self._pan_offset0: Optional[QtCore.QPoint] = None

        # 人工裁决层 QImage（供兼容/绘制）
        self._keep_img: Optional[QtGui.QImage] = None
        self._drop_img: Optional[QtGui.QImage] = None

        # LOD：交互中快速响应，停下 120ms 后再按倍率切换平滑/最近邻
        self._lod_timer = QtCore.QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.setInterval(120)
        self._lod_timer.timeout.connect(self._settle_lod)

    # ---------- 渲染层：把 _scale/_offset 施加到内容项 ----------
    def _apply_transform(self) -> None:
        self._content.setPos(float(self._offset.x()), float(self._offset.y()))
        self._content.setScale(max(1e-6, float(self._scale)))
        self.viewport().update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # 场景矩形贴合控件尺寸（用 self.width/height 而非 viewport()：
        # 后者在未 show()/未派发事件时会滞后，导致取到过期尺寸）
        vw = max(1, self.width())
        vh = max(1, self.height())
        self.setSceneRect(QtCore.QRectF(0, 0, vw, vh))

    # ---------- 对外接口 ----------
    def set_image(self, rgb: np.ndarray, alpha: Optional[np.ndarray] = None,
                  bg: Optional[str] = None, keep_transform: bool = True) -> None:
        """设置显示内容。

        keep_transform: 尺寸不变时保留当前缩放/平移，避免连续处理时视图被重置。
        """
        if bg is not None:
            self._bg = bg
        old_size = None if self._pixmap is None else (self._pixmap.width(), self._pixmap.height())
        self._pixmap = composite_to_pixmap(rgb, alpha, self._bg)
        self._content.set_geometry(self._pixmap.width(), self._pixmap.height())
        new_size = (self._pixmap.width(), self._pixmap.height())
        if not keep_transform or old_size != new_size:
            self.fit_view()
        else:
            self._apply_transform()
        self.viewport().update()

    def set_background(self, bg: str) -> None:
        self._bg = bg
        self.viewport().update()

    def set_brush_radius(self, px: int) -> None:
        """点选笔刷半径（原图像素）。"""
        self._brush_radius = max(1, int(px))
        self.viewport().update()

    def set_paint_mode(self, mode: str) -> None:
        """左键涂抹的语义：'keep'=涂保留（绿），'drop'=涂去除（红）。

        右键始终涂相反色，方便一笔里同时补和删。
        """
        self._paint_mode = "drop" if mode == "drop" else "keep"
        self.viewport().update()

    def paint_mode(self) -> str:
        return self._paint_mode

    def regions(self) -> List[List[Tuple[int, int]]]:
        """返回已闭合区域（若还有足够点的草稿，先把它闭合进来）。"""
        if len(self._draft) >= 3 and self._tool in ("polygon", "lasso"):
            self._regions.append(list(self._draft))
            self._draft = []
            self._emit_selection_changed()
        return [list(r) for r in self._regions]

    def set_override_masks(self, keep: Optional[np.ndarray],
                           drop: Optional[np.ndarray]) -> None:
        """显示人工裁决区（绿=已保留 / 红=已去除）。"""
        self._keep_img = _qimage_from_mask(keep, _COLOR_KEEP) if keep is not None and keep.any() else None
        self._drop_img = _qimage_from_mask(drop, _COLOR_DROP) if drop is not None and drop.any() else None
        self.viewport().update()

    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._drawing = False
        self._painting = False
        self._last_paint = None
        self._start = self._cur = None
        self._roi_draft = []
        self.setCursor(self._cursor_for(tool))
        self._emit_selection_changed()
        self.viewport().update()

    def clear_selection(self) -> None:
        """清空全部未提交的选区（不影响已提交的裁决层，也不影响 ROI）。"""
        self._drawing = False
        self._painting = False
        self._last_paint = None
        self._start = self._cur = None
        self._draft = []
        self._regions = []
        self._fg_pts = []
        self._bg_pts = []
        self._redo = []
        self._emit_selection_changed()
        self.viewport().update()

    # ---------- ROI 处理区 ----------
    def roi_regions(self) -> List[List[Tuple[int, int]]]:
        """返回已框选的 ROI 区域列表（图像坐标）。"""
        return [list(r) for r in self._roi_regions]

    def has_roi(self) -> bool:
        return bool(self._roi_regions)

    def clear_roi(self) -> None:
        self._roi_draft = []
        self._roi_regions = []
        self._drawing = False
        self._start = self._cur = None
        self.viewport().update()

    # ---------- 撤销 / 重写 ----------
    def _snapshot(self) -> tuple:
        return (list(self._draft), [list(r) for r in self._regions],
                list(self._fg_pts), list(self._bg_pts))

    def _restore(self, snap: tuple) -> None:
        draft, regions, fg, bg = snap
        self._draft = list(draft)
        self._regions = [list(r) for r in regions]
        self._fg_pts = list(fg)
        self._bg_pts = list(bg)
        self._emit_selection_changed()
        self.viewport().update()

    def _invalidate_redo(self) -> None:
        """产生了新的选区内容 → 旧的重写记录作废（标准 redo 语义）。"""
        if self._redo:
            self._redo = []

    def has_redo(self) -> bool:
        return bool(self._redo)

    def redo(self) -> bool:
        """重写：恢复上一次被撤销掉的选区内容。返回是否真的重写了。

        画布内无可重写内容时返回 False（由主窗口决定是否重写已提交的修改）。
        """
        if not self._redo:
            return False
        snap = self._redo.pop()
        self._restore(snap)
        return True

    def selection_state(self) -> Tuple[int, int, int]:
        return (len(self._regions), len(self._fg_pts), len(self._bg_pts))

    def commit(self) -> bool:
        """提交当前所有区域/种子点。返回是否真的有内容可提交。

        若还有正在画的草稿且点数足够，先把它闭合进来。
        """
        if len(self._draft) >= 3 and self._tool in ("polygon", "lasso"):
            self._regions.append(list(self._draft))
            self._draft = []
        if not self._regions and not self._fg_pts and not self._bg_pts:
            return False
        self.selectionCommitted.emit([list(r) for r in self._regions],
                                     list(self._fg_pts), list(self._bg_pts))
        return True

    def has_pending(self) -> bool:
        """画布上是否还有未提交的草稿/区域/种子点（不触发任何信号）。"""
        return bool(self._draft or self._regions or self._fg_pts or self._bg_pts)

    def take_selection(self) -> Tuple[List[List[Tuple[int, int]]],
                                     List[Tuple[int, int]],
                                     List[Tuple[int, int]]]:
        """取走全部未提交的区域/涂抹点并清空画布（不触发 selectionCommitted）。

        M13 新增：「开始处理」前调用，把还没按 Enter 的涂抹并进裁决层——
        修复"画完点选直接点开始处理，点选不生效、圆点一直残留在画面上"的问题。
        语义与 commit() 一致：fg_pts=保留（绿）、bg_pts=去除（红）。
        """
        if len(self._draft) >= 3 and self._tool in ("polygon", "lasso"):
            self._regions.append(list(self._draft))
            self._draft = []
        regions = [list(r) for r in self._regions]
        fg_pts = list(self._fg_pts)
        bg_pts = list(self._bg_pts)
        self.clear_selection()
        return regions, fg_pts, bg_pts

    def undo(self) -> bool:
        """逐级撤销。返回 True 表示画布内确有内容被撤销。

        顺序：① 正在画的草稿点 → ② 已闭合的区域 → ③ 都没有则请求主窗口
        撤销上一次已提交的修改（undoRequested）。
        每次撤销前先把当前选区状态压入重写栈（redo）。
        """
        if not (self._draft or self._regions or self._fg_pts or self._bg_pts):
            self.undoRequested.emit()
            return False
        self._redo.append(self._snapshot())      # 记录撤销前状态，供重写恢复
        if self._draft:
            self._draft.pop()
            self._emit_selection_changed()
            self.viewport().update()
            return True
        if self._regions:
            # 重新变回草稿，方便接着改而不是整块丢掉
            self._draft = list(self._regions.pop())
            self._emit_selection_changed()
            self.viewport().update()
            return True
        if self._fg_pts:
            self._fg_pts.pop()
            self._emit_selection_changed()
            self.viewport().update()
            return True
        self._bg_pts.pop()
        self._emit_selection_changed()
        self.viewport().update()
        return True

    # ---------- 视图 ----------
    def fit_view(self) -> None:
        """缩放并居中，使整图刚好放进画布。"""
        if self._pixmap is None:
            return
        cw, ch = max(1, self.width()), max(1, self.height())
        iw, ih = max(1, self._pixmap.width()), max(1, self._pixmap.height())
        s = min(cw / iw, ch / ih)
        self._scale = max(_MIN_SCALE, min(_MAX_SCALE, s))
        self._offset = QtCore.QPoint(int((cw - iw * self._scale) / 2),
                                     int((ch - ih * self._scale) / 2))
        self._apply_transform()

    def zoom_by(self, factor: float, center: Optional[QtCore.QPointF] = None) -> None:
        """按倍率缩放并保持 center 处的图像点不动（center 缺省为视口中心）。"""
        if self._pixmap is None:
            return
        old = self._scale
        self._scale = max(_MIN_SCALE, min(_MAX_SCALE, self._scale * factor))
        if self._scale == old:
            return
        if center is None:
            center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        ox = center.x() - (center.x() - self._offset.x()) * (self._scale / old)
        oy = center.y() - (center.y() - self._offset.y()) * (self._scale / old)
        self._offset = QtCore.QPoint(int(ox), int(oy))
        # 交互中用最近邻（响应快），停下 120ms 后再按倍率切平滑（轻量 LOD）
        self._content.setScale(self._scale)
        self._content.setPos(float(self._offset.x()), float(self._offset.y()))
        self._lod_timer.start()
        self.viewport().update()

    def _sync_transform_mode(self) -> None:
        """放大（看清单个像素）用最近邻，缩小时用平滑（防锯齿）。"""
        # 内容项由 _ContentItem 绘制；这里控制图像合成项/内容的平滑策略。
        # QGraphicsView 层面用 renderHint 控制，内容项本身用最近邻/平滑二选一。
        smooth = self._scale < 1.0
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, smooth)
        self.viewport().update()

    def _settle_lod(self) -> None:
        self._sync_transform_mode()

    def _cursor_for(self, tool: str):
        if tool in ("rect", "lasso", "roi"):
            return QtCore.Qt.CrossCursor
        if tool in ("fg", "polygon"):
            return QtCore.Qt.PointingHandCursor
        return QtCore.Qt.ArrowCursor

    def _emit_selection_changed(self) -> None:
        self.selectionChanged.emit(len(self._regions), len(self._fg_pts), len(self._bg_pts))

    # ---------- 坐标换算（语义与旧版一致）----------
    def _to_image(self, pos: QtCore.QPoint) -> Tuple[int, int]:
        """窗口坐标 → 原图像素坐标（受缩放平移影响，故拖动画面不影响已有点）。"""
        if self._pixmap is None:
            return (0, 0)
        x = int((pos.x() - self._offset.x()) / self._scale)
        y = int((pos.y() - self._offset.y()) / self._scale)
        x = max(0, min(self._pixmap.width() - 1, x))
        y = max(0, min(self._pixmap.height() - 1, y))
        return (x, y)

    def _to_widget(self, img_pt) -> QtCore.QPointF:
        return QtCore.QPointF(img_pt[0] * self._scale + self._offset.x(),
                              img_pt[1] * self._scale + self._offset.y())

    # ---------- 事件 ----------
    def wheelEvent(self, ev):
        if self._pixmap is None:
            return
        factor = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
        self.zoom_by(factor, ev.position())
        ev.accept()

    def mousePressEvent(self, ev):
        if self._pixmap is None:
            return
        pos = ev.position().toPoint()

        # 平移：中键，或 Ctrl + 左键（触摸板友好）
        if ev.button() == QtCore.Qt.MiddleButton or \
                (ev.button() == QtCore.Qt.LeftButton
                 and ev.modifiers() & QtCore.Qt.ControlModifier):
            self._panning = True
            self._pan_start = pos
            self._pan_offset0 = QtCore.QPoint(self._offset)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            ev.accept()
            return

        if self._tool == "none":
            return

        p = self._to_image(pos)
        if ev.button() == QtCore.Qt.LeftButton:
            if self._tool == "rect":
                self._drawing = True
                self._start = p
                self._cur = p
            elif self._tool == "roi":
                self._drawing = True
                self._start = p
                self._cur = p
            elif self._tool == "fg":
                self._painting = True
                self._paint_at(p, self._paint_mode)
            elif self._tool == "polygon":
                self._draft.append(p)
                self._invalidate_redo()
            elif self._tool == "lasso":
                self._drawing = True
                self._draft = [p]
        elif ev.button() == QtCore.Qt.RightButton:
            if self._tool == "fg":
                # 右键涂相反色：作用=保留时右键涂去除，反之亦然
                self._painting = True
                self._paint_at(p, "drop" if self._paint_mode == "keep" else "keep")
            elif self._tool in ("polygon", "lasso", "rect"):
                self._close_draft()
        self.viewport().update()

    def _paint_at(self, p: Tuple[int, int], mode: str) -> None:
        """加入一个涂抹点。mode='keep' → 绿点(_fg_pts)，'drop' → 红点(_bg_pts)。"""
        self._invalidate_redo()
        lst = self._fg_pts if mode == "keep" else self._bg_pts
        lst.append((int(p[0]), int(p[1])))
        self._last_paint = (int(p[0]), int(p[1]))
        self._emit_selection_changed()

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        if self._panning and self._pan_start is not None:
            self._offset = QtCore.QPoint(
                self._pan_offset0.x() + (pos.x() - self._pan_start.x()),
                self._pan_offset0.y() + (pos.y() - self._pan_start.y()))
            self._apply_transform()
            return
        if self._painting and self._tool == "fg":
            # 连续涂抹：与上一个点拉开半个笔刷才记一个，避免点列爆炸
            p = self._to_image(pos)
            step = max(1, self._brush_radius // 3)
            if self._last_paint is None or \
                    abs(p[0] - self._last_paint[0]) + abs(p[1] - self._last_paint[1]) >= step:
                mode = self._paint_mode
                if ev.buttons() & QtCore.Qt.RightButton and not (ev.buttons() & QtCore.Qt.LeftButton):
                    mode = "drop" if mode == "keep" else "keep"
                self._paint_at(p, mode)
                self.viewport().update()
            return
        if self._drawing and self._tool in ("rect", "lasso", "roi"):
            self._cur = self._to_image(pos)
            if self._tool == "lasso":
                self._draft.append(self._cur)
            self.viewport().update()

    def mouseReleaseEvent(self, ev):
        if self._panning and (
                ev.button() == QtCore.Qt.MiddleButton or ev.button() == QtCore.Qt.LeftButton):
            self._panning = False
            self._pan_start = None
            self.setCursor(self._cursor_for(self._tool))
            return
        if self._painting and ev.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self._painting = False
            self._last_paint = None
            return
        if ev.button() == QtCore.Qt.LeftButton and self._drawing:
            if self._tool in ("rect", "roi"):
                x0, y0 = self._start
                x1, y1 = self._cur
                x, y = min(x0, x1), min(y0, y1)
                w, h = abs(x1 - x0), abs(y1 - y0)
                if w >= 2 and h >= 2:
                    rect = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    if self._tool == "roi":
                        self._roi_regions = [rect]      # ROI 只保留最新一个框
                    else:
                        self._invalidate_redo()
                        self._regions.append(rect)
                        self._emit_selection_changed()
            elif self._tool == "lasso":
                self._close_draft()
            self._drawing = False
            self._start = self._cur = None
            self.viewport().update()

    def mouseDoubleClickEvent(self, ev):
        if self._tool in ("polygon", "lasso"):
            self._close_draft()

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.commit()
        elif ev.key() == QtCore.Qt.Key_Escape:
            self.clear_selection()
        elif ev.key() == QtCore.Qt.Key_Z and ev.modifiers() & QtCore.Qt.ControlModifier:
            self.undo()
        elif ev.key() == QtCore.Qt.Key_Y and ev.modifiers() & QtCore.Qt.ControlModifier:
            self.redo()
        elif ev.key() == QtCore.Qt.Key_0 and ev.modifiers() & QtCore.Qt.ControlModifier:
            self.fit_view()
        else:
            super().keyPressEvent(ev)

    def _close_draft(self) -> None:
        if len(self._draft) >= 3:
            self._invalidate_redo()
            self._regions.append(list(self._draft))
            self._emit_selection_changed()
        self._draft = []
        self._drawing = False
        self.viewport().update()

    # ---------- 绘制（在原图像素坐标系里；Qt 按视口裁剪）----------
    def _paint_content(self, painter: QtGui.QPainter) -> None:
        if self._pixmap is None:
            return

        def cosmetic(color: QtGui.QColor, width: float = 2.0,
                     style=QtCore.Qt.SolidLine) -> QtGui.QPen:
            pen = QtGui.QPen(color, width, style)
            pen.setCosmetic(True)      # 线宽不随缩放变化
            return pen

        # 1) 图像本体（透明区已合成到底板）
        painter.drawPixmap(0, 0, self._pixmap)

        # 2) 人工裁决层（半透明覆盖）
        for img in (self._keep_img, self._drop_img):
            if img is not None:
                painter.drawImage(0, 0, img)

        # 3) ROI 处理区（琥珀虚线）
        painter.setPen(cosmetic(_COLOR_ROI, 2.0, QtCore.Qt.DashLine))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(240, 170, 40, 45)))
        for pts in self._roi_regions:
            painter.drawPolygon(self._poly(pts))
        if self._tool == "roi" and self._drawing and self._start and self._cur:
            r = QtCore.QRectF(QtCore.QPointF(*self._start),
                              QtCore.QPointF(*self._cur)).normalized()
            painter.drawRect(r)

        if self._tool == "none" and not self._regions \
                and not self._fg_pts and not self._bg_pts and not self._draft:
            return

        # 4) 已闭合的区域
        for pts in self._regions:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(46, 204, 113, 60)))
            painter.setPen(cosmetic(_COLOR_REGION, 2.0))
            painter.drawPolygon(self._poly(pts))

        # 5) 正在画的草稿
        if self._tool == "rect" and self._start and self._cur:
            a = QtCore.QPointF(*self._start)
            b = QtCore.QPointF(*self._cur)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(120, 220, 160, 50)))
            painter.setPen(cosmetic(_COLOR_DRAFT, 2.0))
            painter.drawRect(QtCore.QRectF(a, b).normalized())
        elif self._tool in ("polygon", "lasso") and self._draft:
            painter.setPen(cosmetic(_COLOR_DRAFT, 2.0))
            pts = [QtCore.QPointF(float(x), float(y)) for x, y in self._draft]
            painter.drawPolyline(QtGui.QPolygonF(pts))
            # 顶点圆点：cosmetic 画笔（屏幕直径恒定 ≈3px），不随缩放放大 ——
            # 描边需放大到看清单个像素，圆点若随之变大会盖住要描的轮廓。
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(vertex_dot_pen(_COLOR_DRAFT))
            for p in pts:
                painter.drawPoint(p)

        # 6) 点选笔刷涂抹（半透明填充，重叠处自然连成一片）
        if self._fg_pts or self._bg_pts:
            # 半径严格按用户设定（最小 1），不再强制下限 3 ——
            # 修复"设成 2 却仍然很大"：旧代码 max(3, radius) 把 2 顶成了 3。
            r = float(max(1, self._brush_radius))
            painter.setPen(QtCore.Qt.NoPen)
            for pt, color in [(p, _COLOR_FG) for p in self._fg_pts] + \
                             [(p, _COLOR_BG) for p in self._bg_pts]:
                c = QtGui.QColor(color)
                c.setAlpha(150)
                painter.setBrush(QtGui.QBrush(c))
                painter.drawEllipse(QtCore.QPointF(float(pt[0]), float(pt[1])), r, r)

    @staticmethod
    def _poly(pts) -> QtGui.QPolygonF:
        return QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in pts])

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        super().drawForeground(painter, rect)
        if self._pixmap is not None:
            return
        painter.save()
        painter.resetTransform()
        painter.setPen(QtGui.QColor(150, 150, 155))
        painter.drawText(self.viewport().rect(), QtCore.Qt.AlignCenter,
                         "拖入图片或点击「打开」")
        painter.restore()
