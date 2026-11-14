#!/usr/bin/env python3
"""Assemble the app's model bundle (app/src/main/assets/models) from mobile_app/models.

Files are copied under canonical names (yolo.onnx, script_id.onnx, parseq_hindi.onnx,
parseq_english.onnx) from the chosen variants, and manifest.json gets an "app" block
(default script-ID mode + which variant each file came from). --no-script-id leaves the 450 MB
script-ID model out (app then always uses dual mode).
"""
import argparse, json, shutil
from pathlib import Path

M = Path("/DATA3/detect/faisal/mobile_app/models")
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="/DATA3/detect/faisal/mobile_app/android/app/src/main/assets/models")
ap.add_argument("--yolo", default="yolo.onnx")
ap.add_argument("--script-id", default="script_id.onnx")
ap.add_argument("--script-id-size", type=int, default=None, help="input size of --script-id (default: manifest, 288 for V2-L)")
ap.add_argument("--no-script-id", action="store_true")
ap.add_argument("--parseq-suffix", default="_int8")
ap.add_argument("--si-mode", default="model", choices=("model", "dual"))
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
files = {"yolo.onnx": a.yolo, "parseq_hindi.onnx": f"parseq_hindi{a.parseq_suffix}.onnx",
         "parseq_english.onnx": f"parseq_english{a.parseq_suffix}.onnx"}
if not a.no_script_id:
    files["script_id.onnx"] = a.script_id
for dst, src in files.items():
    shutil.copy2(M / src, out / dst)
m = json.loads((M / "manifest.json").read_text())
m["parseq"] = {k: v for k, v in m["parseq"].items() if k in ("hindi", "english")}
if a.script_id_size:
    m["script_id"]["size"] = a.script_id_size
m["app"] = {"si_mode": "dual" if a.no_script_id else a.si_mode, "sources": files}
(out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
for f in sorted(out.iterdir()):
    print(f"{f.name:22s} {f.stat().st_size / 2**20:8.1f} MB")
