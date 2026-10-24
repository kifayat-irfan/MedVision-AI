#!/usr/bin/env python3
"""Download BUSI images via HTTP (threaded). Skips valid existing files."""
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import threading

OUT = Path.home() / "workspace/github-fix/MedVision-AI/data/ultrasound/busi"
REPO = "theshoaibme/panorgan-busi-breast-ultrasound"

req = urllib.request.Request(f"https://huggingface.co/api/datasets/{REPO}")
with urllib.request.urlopen(req, timeout=60) as r:
    d = json.load(r)
files = [s["rfilename"] for s in d.get("siblings", [])
         if s["rfilename"].endswith(".png") and "_mask" not in s["rfilename"]]
print(f"Found {len(files)} images", flush=True)

base = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
lock = threading.Lock()
done, failed, skipped = [0], [0], [0]


def valid_png(p):
    if not p.exists() or p.stat().st_size < 1000:
        return False
    try:
        with open(p, "rb") as f:
            sig = f.read(8)
        return sig[:4] == b"\x89PNG"
    except Exception:
        return False


def dl(f):
    dest = OUT / f
    if valid_png(dest):
        with lock:
            skipped[0] += 1
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(base + quote(f, safe="/"), str(dest))
        if not valid_png(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError("not a valid PNG")
        with lock:
            done[0] += 1
    except Exception as e:
        with lock:
            failed[0] += 1
            print(f"FAIL {f}: {str(e)[:60]}", flush=True)
    with lock:
        n = done[0] + failed[0] + skipped[0]
        if n % 100 == 0:
            print(f"progress: {n}/{len(files)} (new={done[0]} fail={failed[0]} skip={skipped[0]})",
                  flush=True)


with ThreadPoolExecutor(max_workers=6) as ex:
    list(ex.map(dl, files))
print(f"DONE: new={done[0]} failed={failed[0]} skipped={skipped[0]}")
