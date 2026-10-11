"""LucentCut 桌面应用启动入口。

用法：python -m src.app  （或 python src/app.py）
"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6 import QtGui, QtWidgets
    from src.ui.gui import MainWindow
    from src.ui.resources import app_icon_path

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("LucentCut")
    # 窗口/任务栏图标：Qt 不会自动继承 exe 的内嵌图标，必须显式设置。
    # 打包后从 sys._MEIPASS/assets/logo/lucentcut.ico 取（见 build.spec 的 datas）。
    icon = QtGui.QIcon(app_icon_path())
    if not icon.isNull():
        app.setWindowIcon(icon)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
