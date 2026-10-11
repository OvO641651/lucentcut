"""运行时资源路径解析（开发态 vs PyInstaller 冻结态）。

- 开发态：以项目根为基准（本文件位于 src/ui/ 下，上溯三级）。
- 冻结态：以 PyInstaller 解包目录 sys._MEIPASS 为基准。

配合 build.spec 中的 datas 把 assets/logo/lucentcut.ico 打进 exe，
这样窗口/任务栏图标在打包后依然能找到图标文件。
"""

from __future__ import annotations

import os
import sys


def resource_path(rel: str) -> str:
    """返回随包资源在运行时的绝对路径。"""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:  # 开发态
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, rel)


def app_icon_path() -> str:
    """应用窗口/任务栏图标（多尺寸 ico）。"""
    return resource_path(os.path.join("assets", "logo", "lucentcut.ico"))
