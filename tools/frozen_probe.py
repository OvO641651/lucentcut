"""打包后自检探针：在冻结环境里验证 AI 是否真的可用。

仅在排查打包问题时使用，不属于交付代码。
"""
from __future__ import annotations

import json
import os
import sys
import traceback

result = {"pyiver": sys.version.split()[0]}
try:
    frozen = getattr(sys, "_MEIPASS", None)
    result["frozen"] = bool(frozen)
    result["meipass"] = frozen or ""

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.core import ai_segment, io, pipeline

    ok, why = ai_segment.availability_detail()
    result["ai_available"] = ok
    result["ai_reason"] = why
    result["base_dir"] = ai_segment._base_dir()
    result["model_path"] = ai_segment._find_model() or ""

    if ok:
        import numpy as np
        from PIL import Image

        # 造一张测试图：白底上一个红色方块（主体），验证 AI 能把它抠出来
        img = np.full((320, 320, 3), 255, np.uint8)
        img[80:240, 80:240] = (40, 40, 220)
        a = ai_segment.predict_alpha(img)
        inside = float(np.mean(a[100:220, 100:220] >= 128))
        outside = float(np.mean(a[0:40, 0:40] >= 128))
        result["ai_inside_keep"] = round(inside, 4)
        result["ai_outside_keep"] = round(outside, 4)
        result["ai_ok"] = inside > 0.9 and outside < 0.05

        # 完整混合引擎跑一遍
        rgb_out, alpha_out, mode, rescued, reused, warns = pipeline.process_array(
            img, None, engine="hybrid")
        result["hybrid_mode"] = mode
        result["hybrid_shape"] = list(alpha_out.shape)
        result["hybrid_ok"] = list(alpha_out.shape) == [320, 320, 3][:0] + [320, 320]
except Exception:
    result["error"] = traceback.format_exc()

out = os.environ.get("PROBE_OUT", "probe_result.json")
try:
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
except Exception:
    traceback.print_exc()
print(json.dumps(result, ensure_ascii=False, indent=2))
