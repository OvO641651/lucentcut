"""模型下载工具（M12）。

把可选的高质量抠图模型下载到 src/models/models/ 下，供 LucentCut 的
「精细抠图」引擎使用。支持断点续传与完整性校验，可重复运行。

用法：
    .venv/Scripts/python.exe tools/download_models.py                # 下载全部
    .venv/Scripts/python.exe tools/download_models.py birefnet_tiny  # 只下一个

说明：模型体积较大且走公网，慢的话可重复执行本脚本续传。
"""

from __future__ import annotations

import os
import socket
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(ROOT, "src", "models", "models")

# key -> (相对路径, 下载 URL, 说明)
CATALOG = {
    "birefnet_tiny": (
        "birefnet/birefnet_tiny.onnx",
        "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/"
        "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "BiRefNet swin-tiny（MIT，1024×1024，推荐）",
    ),
    "isnet": (
        "isnet/isnet.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "isnet-general-use.onnx",
        "ISNet general-use（1024×1024，备用）",
    ),
    "birefnet_lite2k": (
        "birefnet/birefnet_lite2k.onnx",
        "https://huggingface.co/zurg70/birefnet-desktop-models/resolve/main/"
        "birefnet_general_lite2k.onnx",
        "BiRefNet lite-2K（MIT，2560×1440，慢）",
    ),
}


def download(key: str, max_retries: int = 20) -> bool:
    rel, url, desc = CATALOG[key]
    dst = os.path.join(MODEL_ROOT, rel)
    tmp = dst + ".part"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {key} 已存在：{dst}")
        return True
    if not os.path.exists(tmp) and os.path.exists(dst):
        os.rename(dst, tmp)

    try:
        head = urllib.request.Request(url, method="HEAD",
                                      headers={"User-Agent": "Mozilla/5.0"})
        total = int(urllib.request.urlopen(head).headers.get("Content-Length", 0))
    except Exception:  # noqa: BLE001
        total = 0

    print(f"[get ] {key} — {desc}（断点续传，最多重试 {max_retries} 次）")
    t0 = time.time()
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            last = time.time()
            with urllib.request.urlopen(req) as r, open(tmp, "ab" if have else "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if time.time() - last > 15:
                        speed = have / 1e6 / max(time.time() - t0, 1e-3)
                        print(f"       {have / 1e6:.1f}/{total / 1e6:.1f} MB  "
                              f"{speed:.2f} MB/s (retry {attempt})", flush=True)
                        last = time.time()
        except Exception as e:  # noqa: BLE001
            print(f"       [drop] {type(e).__name__}: {str(e)[:60]} — 准备续传")
            time.sleep(1.5)

        if total and have >= total - 65536:
            os.replace(tmp, dst)
            print(f"[done] {dst}  {os.path.getsize(dst)} bytes  "
                  f"用时 {time.time() - t0:.0f}s")
            return True
    print(f"[fail] {key} 多次重试仍不完整（{have}/{total}），请稍后重跑本脚本续传")
    return False


def main() -> int:
    socket.setdefaulttimeout(60)
    keys = sys.argv[1:] or list(CATALOG)
    ok = all(download(k) for k in keys if k in CATALOG)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
