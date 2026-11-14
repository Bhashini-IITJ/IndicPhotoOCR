#!/usr/bin/env python3
"""Assemble the all-language app bundle: the Hindi+English bundle (YOLO fp32, EfficientNet-B0 script ID)
plus one int8 PARSeq reader per script-ID class, from mobile_app/models_all.

Example:
  python make_bundle_all.py --out ../bundles/b0_int8_all
"""
import argparse, json, shutil
from pathlib import Path

MA = Path("/DATA3/detect/faisal/mobile_app")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", default=str(MA / "bundles/b0_int8"), help="bundle with yolo.onnx + script_id.onnx")
ap.add_argument("--readers", default=str(MA / "models_all"))
ap.add_argument("--parseq-suffix", default="_int8")
ap.add_argument("--out", required=True)
a = ap.parse_args()
base, rd, out = Path(a.base), Path(a.readers), Path(a.out)
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
m = json.loads((base / "manifest.json").read_text())
mr = json.loads((rd / "manifest.json").read_text())
for f in ("yolo.onnx", "script_id.onnx"):
    shutil.copy2(base / f, out / f)
m["parseq"] = {}
for lang in m["script_id"]["classes"]:
    shutil.copy2((rd / f"parseq_{lang}{a.parseq_suffix}.onnx").resolve(), out / f"parseq_{lang}.onnx")
    m["parseq"][lang] = dict(mr["parseq"][lang], file=f"parseq_{lang}.onnx")
m.setdefault("app", {})["readers"] = {"source": str(rd), "suffix": a.parseq_suffix}
(out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
total = 0
for f in sorted(out.iterdir()):
    total += f.stat().st_size
    print(f"{f.name:24s} {f.stat().st_size / 2**20:8.1f} MB")
print(f"{'total':24s} {total / 2**20:8.1f} MB")
