"""预览画布组件（M2 + M7 + M9 交互增强）。

职责：
- 显示原图 / 结果图，支持底板切换（棋盘格 / 白 / 黑 / 红）
- 视图操作：滚轮缩放、中键（或 Ctrl+左键）拖动平移、适应窗口
- 手动抠图交互：矩形框选、点选（前景/背景）、描边多边形、套索自由曲线
- 一次可描**多个互不相连的区域**，支持逐级撤销

绘制工具（tool）：
- "none"     仅查看，可缩放/平移
- "rect"     拖拽画矩形框（松开后成为一个区域）
- "fg"       点选**笔刷**：按住左键涂抹「当前作用」色（保留/去除），按住右键涂反色；
             可连续拖动涂出一整片，笔刷大小见「点选笔刷半径」
- "polygon"  左键点顶点，双击 / 右键 / 回车闭合为一个区域
- "lasso"    按住左键拖拽画自由曲线，松开闭合为一个区域

区域（region）统一用「多边形点列」表示，闭合后进入区域列表，
可继续画下一个区域；按 Enter 或点「应用选区」统一提交。

视图操作（M9）：
- 滚轮          以光标为中心缩放
- 中键拖动      平移画布（也可 Ctrl + 左键拖动，方便没有中键的触摸板）
- Ctrl+0 / 双击中键   适应窗口
平移与缩放只改变显示变换，已描出的点保存在**原图坐标**下，
因此拖动画面不会影响任何已完成的描边内容。
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


def _mask_to_qimage(mask: np.ndarray, color: QtGui.QColor, alpha: int = 96) -> QtGui.QImage:
    """把二值掩码转成半透明彩色 QImage（用于绘制人工裁决区）。"""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    m = mask > 0
    out[m, 0] = color.red()
    out[m, 1] = color.green()
    out[m, 2] = color.blue()
    out[m, 3] = alpha
    return QtGui.QImage(out.tobytes(), w, h, QtGui.QImage.Format_RGBA8888).copy()


class Canvas(QtWidgets.QWidget):
    """预览画布：视图变换 + 多区域交互式选区绘制。"""

    # 选区变化（用于更新状态栏与按钮可用性）
    selectionChanged = QtCore.Signal(int, int, int)        # (区域数, 前景点数, 背景点数)
    # 用户按下 Enter / 点「应用选区」：一次性提交全部区域与种子点
    selectionCommitted = QtCore.Signal(list, list, list)   # (regions, fg_pts, bg_pts)
    # 画布内无可撤销内容时请求主窗口回退上一次已提交的修改
    undoRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._bg = "checker"
        self._tool = "none"
        self._scale = 1.0
        self._offset = QtCore.QPoint(0, 0)

        # 交互状态
        self._drawing = False              # 矩形拖拽 / 套索拖拽中
        self._start: Optional[Tuple[int, int]] = None
        self._cur: Optional[Tuple[int, int]] = None
        self._draft: List[Tuple[int, int]] = []        # 正在画的区域的点
        self._regions: List[List[Tuple[int, int]]] = []  # 已闭合的区域
        self._fg_pts: List[Tuple[int, int]] = []       # 点选：当前作用色的涂抹点
        self._bg_pts: List[Tuple[int, int]] = []       # 点选：反色的涂抹点

        # 点选笔刷（直接裁决，不走 GrabCut）
        self._brush_radius = 12            # 按原图像素计
        self._paint_mode = "keep"          # 左键涂什么颜色：keep / drop
        self._painting = False
        self._last_paint: Optional[Tuple[int, int]] = None

        # 视图状态
        self._panning = False
        self._pan_start: Optional[QtCore.QPoint] = None
        self._pan_offset0: Optional[QtCore.QPoint] = None

        # 人工裁决层可视化
        self._keep_img: Optional[QtGui.QImage] = None
        self._drop_img: Optional[QtGui.QImage] = None

        self.setMouseTracking(True)
        self.setMinimumSize(320, 320)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

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
        new_size = (self._pixmap.width(), self._pixmap.height())
        if not keep_transform or old_size != new_size:
            self.fit_view()
        self.update()

    def set_background(self, bg: str) -> None:
        self._bg = bg
        self.update()

    def set_brush_radius(self, px: int) -> None:
        """点选笔刷半径（原图像素）。"""
        self._brush_radius = max(1, int(px))
        self.update()

    def set_paint_mode(self, mode: str) -> None:
        """左键涂抹的语义：'keep'=涂保留（绿），'drop'=涂去除（红）。

        右键始终涂相反色，方便一笔里同时补和删。
        """
        self._paint_mode = "drop" if mode == "drop" else "keep"
        self.update()

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
        self._keep_img = _mask_to_qimage(keep, _COLOR_KEEP) if keep is not None and keep.any() else None
        self._drop_img = _mask_to_qimage(drop, _COLOR_DROP) if drop is not None and drop.any() else None
        self.update()

    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._drawing = False
        self._painting = False
        self._last_paint = None
        self._start = self._cur = None
        self.setCursor(self._cursor_for(tool))
        self._emit_selection_changed()
        self.update()

    def clear_selection(self) -> None:
        """清空全部未提交的选区（不影响已提交的裁决层）。"""
        self._drawing = False
        self._painting = False
        self._last_paint = None
        self._start = self._cur = None
        self._draft = []
        self._regions = []
        self._fg_pts = []
        self._bg_pts = []
        self._emit_selection_changed()
        self.update()

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
        """
        if self._draft:
            self._draft.pop()
            self._emit_selection_changed()
            self.update()
            return True
        if self._regions:
            # 重新变回草稿，方便接着改而不是整块丢掉
            self._draft = list(self._regions.pop())
            self._emit_selection_changed()
            self.update()
            return True
        if self._fg_pts:
            self._fg_pts.pop()
            self._emit_selection_changed()
            self.update()
            return True
        if self._bg_pts:
            self._bg_pts.pop()
            self._emit_selection_changed()
            self.update()
            return True
        self.undoRequested.emit()
        return False

    # ---------- 视图 ----------
    def fit_view(self) -> None:
        """缩放并居中，使整图刚好放进画布。"""
        if self._pixmap is None:
            return
        cw, ch = max(1, self.width()), max(1, self.height())
        iw, ih = max(1, self._pixmap.width()), max(1, self._pixmap.height())
        s = min(cw / iw, ch / ih)
        self._scale = max(0.02, min(16.0, s))
        self._offset = QtCore.QPoint(int((cw - iw * self._scale) / 2),
                                     int((ch - ih * self._scale) / 2))
        self.update()

    def zoom_by(self, factor: float, center: Optional[QtCore.QPointF] = None) -> None:
        if self._pixmap is None:
            return
        old = self._scale
        self._scale = max(0.02, min(32.0, self._scale * factor))
        if center is None:
            center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        self._offset.setX(int(center.x() - (center.x() - self._offset.x()) * (self._scale / old)))
        self._offset.setY(int(center.y() - (center.y() - self._offset.y()) * (self._scale / old)))
        self.update()

    def _cursor_for(self, tool: str):
        if tool in ("rect", "lasso"):
            return QtCore.Qt.CrossCursor
        if tool in ("fg", "polygon"):
            return QtCore.Qt.PointingHandCursor
        return QtCore.Qt.ArrowCursor

    def _emit_selection_changed(self) -> None:
        self.selectionChanged.emit(len(self._regions), len(self._fg_pts), len(self._bg_pts))

    # ---------- 坐标换算 ----------
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
            return

        if self._tool == "none":
            return

        p = self._to_image(pos)
        if ev.button() == QtCore.Qt.LeftButton:
            if self._tool == "rect":
                self._drawing = True
                self._start = p
                self._cur = p
            elif self._tool == "fg":
                self._painting = True
                self._paint_at(p, self._paint_mode)
            elif self._tool == "polygon":
                self._draft.append(p)
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
        self.update()

    def _paint_at(self, p: Tuple[int, int], mode: str) -> None:
        """加入一个涂抹点。mode='keep' → 绿点(_fg_pts)，'drop' → 红点(_bg_pts)。"""
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
            self.update()
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
                self.update()
            return
        if self._drawing and self._tool in ("rect", "lasso"):
            self._cur = self._to_image(pos)
            if self._tool == "lasso":
                self._draft.append(self._cur)
            self.update()

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
            if self._tool == "rect":
                x0, y0 = self._start
                x1, y1 = self._cur
                x, y = min(x0, x1), min(y0, y1)
                w, h = abs(x1 - x0), abs(y1 - y0)
                if w >= 2 and h >= 2:
                    self._regions.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
                    self._emit_selection_changed()
            elif self._tool == "lasso":
                self._close_draft()
            self._drawing = False
            self._start = self._cur = None
            self.update()

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
        elif ev.key() == QtCore.Qt.Key_0 and ev.modifiers() & QtCore.Qt.ControlModifier:
            self.fit_view()
        else:
            super().keyPressEvent(ev)

    def _close_draft(self) -> None:
        if len(self._draft) >= 3:
            self._regions.append(list(self._draft))
            self._emit_selection_changed()
        self._draft = []
        self._drawing = False
        self.update()

    # ---------- 绘制 ----------
    def paintEvent(self, ev):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(35, 35, 38))

        if self._pixmap is None:
            painter.setPen(QtGui.QColor(150, 150, 155))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "拖入图片或点击「打开」")
            painter.end()
            return

        target_w = max(1, int(self._pixmap.width() * self._scale))
        target_h = max(1, int(self._pixmap.height() * self._scale))
        mode = QtCore.Qt.FastTransformation if self._scale >= 1.0 \
            else QtCore.Qt.SmoothTransformation
        scaled = self._pixmap.scaled(target_w, target_h, QtCore.Qt.KeepAspectRatio, mode)
        painter.drawPixmap(self._offset.x(), self._offset.y(), scaled)

        # 人工裁决层（半透明覆盖）
        for img in (self._keep_img, self._drop_img):
            if img is not None:
                painter.drawImage(
                    QtCore.QRectF(self._offset.x(), self._offset.y(), target_w, target_h), img)

        if self._tool == "none":
            painter.end()
            return

        # 已闭合的区域
        for pts in self._regions:
            poly = QtGui.QPolygonF([self._to_widget(p) for p in pts])
            painter.setBrush(QtGui.QBrush(QtGui.QColor(46, 204, 113, 60)))
            painter.setPen(QtGui.QPen(_COLOR_REGION, 2))
            painter.drawPolygon(poly)

        # 正在画的草稿
        if self._tool == "rect" and self._start and self._cur:
            a = self._to_widget(self._start)
            b = self._to_widget(self._cur)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(120, 220, 160, 50)))
            painter.setPen(QtGui.QPen(_COLOR_DRAFT, 2))
            painter.drawRect(QtCore.QRectF(a, b).normalized())
        elif self._tool in ("polygon", "lasso") and self._draft:
            painter.setPen(QtGui.QPen(_COLOR_DRAFT, 2))
            pts = [self._to_widget(p) for p in self._draft]
            painter.drawPolyline(QtGui.QPolygonF(pts))
            painter.setBrush(QtGui.QBrush(_COLOR_DRAFT))
            painter.setPen(QtCore.Qt.NoPen)
            for p in pts:
                painter.drawEllipse(p, 3.0, 3.0)

        # 点选笔刷涂抹（半透明填充，重叠处自然连成一片）
        if self._fg_pts or self._bg_pts:
            r = max(3.0, self._brush_radius * self._scale)
            painter.setPen(QtCore.Qt.NoPen)
            for pt, color in [(p, _COLOR_FG) for p in self._fg_pts] + \
                             [(p, _COLOR_BG) for p in self._bg_pts]:
                c = QtGui.QColor(color)
                c.setAlpha(150)
                painter.setBrush(QtGui.QBrush(c))
                painter.drawEllipse(self._to_widget(pt), r, r)

        painter.end()
