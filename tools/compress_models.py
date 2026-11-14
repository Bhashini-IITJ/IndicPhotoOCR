#!/usr/bin/env python3
"""Write fp16 and int8 versions of the exported app models next to the fp32 files.

fp16 : onnxconverter-common float16 conversion, inputs/outputs kept fp32 (keep_io_types).
int8 : onnxruntime dynamic quantisation (int8 weights, activations quantised at run time);
       YOLO is left fp16 because its NMS-free raw head is precision sensitive and it is only ~11 MB.
"""
import sys
from pathlib import Path
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/DATA3/detect/faisal/mobile_app/models")
for name in ("yolo", "script_id", "parseq_hindi", "parseq_english"):
    src = D / f"{name}.onnx"
    m = onnx.load(str(src))
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), str(D / f"{name}_fp16.onnx"))
    if name == "yolo":
        (D / "yolo_int8.onnx").write_bytes((D / "yolo_fp16.onnx").read_bytes())
    else:
        quantize_dynamic(str(src), str(D / f"{name}_int8.onnx"), weight_type=QuantType.QInt8)
    print(name, "done", flush=True)
for f in sorted(D.glob("*.onnx")):
    print(f"{f.name:28s} {f.stat().st_size / 2**20:8.1f} MB")
