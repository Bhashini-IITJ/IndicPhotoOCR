#!/usr/bin/env python3
"""All-language variant of reference_pipeline.py (offline app pipeline, ONNX Runtime on CPU).

Same detection, crop, script ID and PARSeq decoding as reference_pipeline.Pipeline, but every one of
the 12 script-ID classes is read by its own PARSeq (as the server does): no word is skipped.
Detector and script ID come from --det-models (the app bundle: YOLO fp32 + EfficientNet-B0);
the PARSeq readers come from --rec-models with file suffix --rec-suffix ("" = fp32, "_int8").
Scores BSTD validation with the official metric (all 10 languages).

Example:
  python reference_pipeline_all.py --rec-suffix _int8 --out ../eval/val_all_int8
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from reference_pipeline import ROOT, Pipeline


class PipelineAll(Pipeline):
    def __init__(self, det_dir, rec_dir, rec_suffix="", threads=8):
        md = json.loads((Path(det_dir) / "manifest.json").read_text())
        mr = json.loads((Path(rec_dir) / "manifest.json").read_text())
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.yolo = ort.InferenceSession(str(Path(det_dir) / md["yolo"]["file"]), so, providers=prov)
        self.sid = ort.InferenceSession(str(Path(det_dir) / md["script_id"]["file"]), so, providers=prov)
        self.ycfg, self.scfg, self.pcfg = md["yolo"], md["script_id"], mr["parseq"]
        self.classes = self.scfg["classes"]
        rf = lambda name: str(Path(rec_dir) / name.replace(".onnx", f"{rec_suffix}.onnx"))
        self.rec = {l: ort.InferenceSession(rf(self.pcfg[l]["file"]), so, providers=prov) for l in self.classes}

    def run(self, bgr, si_mode="model"):
        t = {}; t0 = time.perf_counter()
        boxes, scores = self.detect(bgr); t["detect"] = time.perf_counter() - t0
        crops, keep = [], []
        for j, b in enumerate(boxes):
            c = self.crop(bgr, b)
            if c is not None:
                crops.append(c); keep.append(j)
        t0 = time.perf_counter(); langs = self.script_id(crops) if crops else []; t["script_id"] = time.perf_counter() - t0
        t0 = time.perf_counter(); words = {}
        for lang in self.classes:
            js = [q for q, l in enumerate(langs) if l == lang]
            for q, r in zip(js, self.recognise(lang, [crops[q] for q in js]) if js else []):
                words[q] = r
        t["recognise"] = time.perf_counter() - t0
        out = [{"box": boxes[j].tolist(), "score": float(scores[j]), "si": langs[q],
                "text": words[q][0], "conf": words[q][1]} for q, j in enumerate(keep)]
        return out, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-models", default=str(ROOT / "mobile_app/bundles/b0_int8"))
    ap.add_argument("--rec-models", default=str(ROOT / "mobile_app/models_all"))
    ap.add_argument("--rec-suffix", default="", help='"" (fp32) or "_int8"')
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test", action="store_true", help="needed for --split test (final one-time check)")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th image starting at i (merge with merge_shards)")
    a = ap.parse_args()
    if a.split != "val" and not a.allow_test:
        sys.exit("validation only unless --allow-test")
    sys.path.insert(0, str(ROOT / "scripts"))
    import official_metric_e2e as om
    import run_all_end2end_ensembles as base
    gt = json.load(open(om.GT_PATH, encoding="utf-8"))
    keys = om.split_keys(gt, a.split)[: a.limit or None]
    si, sn = map(int, a.shard.split("/")); keys = keys[si::sn]
    pipe = PipelineAll(a.det_models, a.rec_models, a.rec_suffix, a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    preds = {"__status__": {}}; times = []
    for i, k in enumerate(keys, 1):
        res, t = pipe.run(cv2.imread(base.imgpath(gt[k], str(om.IMAGE_ROOT))))
        preds[k] = {f"polygon_{j}": {"text": r["text"], "si": r["si"], "conf": r["conf"], "box": r["box"], "score": r["score"]}
                    for j, r in enumerate(res)}
        preds["__status__"][k] = "success"; times.append(t)
        if i % 50 == 0:
            print(f"{i}/{len(keys)}", flush=True)
    om.atomic_json(out / "predictions.json", preds)
    if sn > 1:
        return
    table, _ = om.score_system(om.official(), {k: gt[k] for k in keys}, preds)
    summ = {"images": len(keys), "rec_suffix": a.rec_suffix, "threads": a.threads,
            "mean_seconds": {s: round(float(np.mean([t[s] for t in times])), 3) for s in times[0]},
            "official": {l: {m: round(100 * table[l][m], 2) for m in ("precision", "recall", "f1", "wrr", "crr")} for l in table}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"mean_seconds": summ["mean_seconds"], "average": summ["official"]["average"]}, indent=1))


if __name__ == "__main__":
    main()
