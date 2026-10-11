"""冻结环境探针：验证打包后的 exe 里「点选笔刷 / 显示修改区 / 应用选区」是否正常。

exe 是窗口程序（console=False），看不到输出，所以把结果写到 probe_gui_result.json。
用 tools/probe_gui.spec 打包后运行 dist_probe/probe_gui.exe。
"""
import json
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

res = {}
try:
    import numpy as np

    from src.core import ai_segment, interactive
    from src.ui.gui import MainWindow
    from PySide6 import QtWidgets

    ok, why = ai_segment.availability_detail()
    res["ai_available"] = ok
    res["ai_why"] = why

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()

    p = os.path.join("image", "动漫表情包图", "deepseek2.png")
    win._load(p)
    h, w = win._rgb.shape[:2]

    # 默认关闭「显示修改区」
    res["show_ov_default"] = win.show_ov.isChecked()

    # 点选笔刷 + 保留 → 应用选区
    win.action_combo.setCurrentIndex(0)
    win.brush.setValue(8)
    win._set_tool("fg")
    cx, cy = w // 2, h // 2
    win._on_commit([], [(cx, cy)], [])
    if win._worker is not None:
        win._worker.wait()
    app.processEvents()

    res["has_result"] = win._alpha is not None
    res["no_green_overlay"] = win.dst_canvas._keep_img is None
    m = interactive.points_to_mask(win._rgb.shape, [(cx, cy)], 8)
    res["brush_keep_ok"] = bool(np.all(win._alpha[m > 0] == 255))
    res["brush_fg_ratio"] = float((win._alpha >= 128).mean())
    win.close()
except Exception:
    res["error"] = traceback.format_exc()

with open("probe_gui_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
