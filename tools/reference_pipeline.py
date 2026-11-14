#!/usr/bin/env python3
"""Reference implementation of the offline app's OCR pipeline (IndicPhotoOCR 2.0, Hindi + English).

Uses only numpy / OpenCV / onnxruntime on the exported ONNX models, written step by step so the
Kotlin app can mirror it exactly:
  1. YOLO11n (BSTD+MIST): letterbox to long side 1920 (aspect kept, padded to a multiple of 32,
     centred, grey 114), RGB/255 -> raw head (1, 5, N) -> conf >= 0.25 -> NMS IoU 0.7 -> <= 300 boxes
     -> map back to image pixels.
  2. Crop each box from the original image (integer bounding rectangle, as OCR.crop_bbox).
  3. Script ID (EfficientNetV2-L, 12 classes): RGB, resize 288x288 (per axis: area-average when
     shrinking, bilinear when enlarging), ImageNet mean/std.
     App rule: hindi and marathi -> Hindi PARSeq (both Devanagari), english -> English PARSeq,
     any other script -> word skipped (not supported offline).
  4. PARSeq: RGB, resize 32x128 (per axis: area-average when shrinking, bicubic when enlarging), (x/255 - 0.5)/0.5 -> logits (B, 26, C) -> greedy argmax,
     stop at EOS, charset filter -> text, confidence = mean max-prob up to and incl. EOS.
Writes official_metric_e2e-format predictions (+ si, conf) and scores them on BSTD validation
with the official metric (all 10 languages and Devanagari/English separately).
"""
import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

APP_LANGS = {"hindi": "hindi", "marathi": "hindi", "english": "english"}


def resize(img, w, h, up):
    """Separable resize, one axis at a time: area-average on an axis that shrinks (close to PIL /
    torchvision antialiased resize), `up` (bicubic / bilinear) on an axis that grows. Defined per
    axis so the Kotlin app (RgbImage.resizeForModel) implements exactly the same thing."""
    if img.shape[1] != w:
        img = cv2.resize(img, (w, img.shape[0]), interpolation=cv2.INTER_AREA if img.shape[1] > w else up)
    if img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if img.shape[0] > h else up)
    return img


class Pipeline:
    def __init__(self, model_dir, suffix="", threads=8, providers=None):
        m = json.loads((Path(model_dir) / "manifest.json").read_text())
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        prov = providers or ["CPUExecutionProvider"]
        f = lambda name: str(Path(model_dir) / name.replace(".onnx", f"{suffix}.onnx"))
        self.yolo = ort.InferenceSession(f(m["yolo"]["file"]), so, providers=prov)
        self.sid = ort.InferenceSession(f(m["script_id"]["file"]), so, providers=prov)
        self.rec = {l: ort.InferenceSession(f(m["parseq"][l]["file"]), so, providers=prov) for l in ("hindi", "english")}
        self.ycfg, self.scfg, self.pcfg = m["yolo"], m["script_id"], m["parseq"]
        self.classes = self.scfg["classes"]

    # ---- 1. detection
    def detect(self, bgr):
        h, w = bgr.shape[:2]; S = self.ycfg["imgsz"]; st = self.ycfg["stride"]
        r = min(S / h, S / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (S - nw) % st / 2, (S - nh) % st / 2
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)); left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = img[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.yolo.run(None, {self.yolo.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]  # (5, N)
        cx, cy, bw, bh, sc = out
        keep = sc >= self.ycfg["conf"]
        b = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)[keep]; sc = sc[keep]
        idx = self.nms(b, sc, self.ycfg["iou"])[: self.ycfg["max_det"]]
        b, sc = b[idx], sc[idx]
        b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clip(0, w); b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clip(0, h)
        return b, sc

    @staticmethod
    def nms(b, s, thr):
        order = np.argsort(-s, kind="stable"); keep = []
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(b[i, 0], b[order[1:], 0]); yy1 = np.maximum(b[i, 1], b[order[1:], 1])
            xx2 = np.minimum(b[i, 2], b[order[1:], 2]); yy2 = np.minimum(b[i, 3], b[order[1:], 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= thr]
        return np.array(keep, int)

    # ---- 2. crop
    @staticmethod
    def crop(bgr, box):
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]  # np.int32 truncation, as OCR.crop_bbox
        x1, x2 = np.clip([x1, x2], 0, w - 1); y1, y2 = np.clip([y1, y2], 0, h - 1)
        if x2 < x1 or y2 < y1:
            return None
        c = bgr[y1:y2 + 1, x1:x2 + 1]
        return c if c.size else None

    # ---- 3. script id
    def script_id(self, crops):
        m, s = np.array(self.scfg["mean"], np.float32), np.array(self.scfg["std"], np.float32)
        S = self.scfg["size"]; out = []
        for i in range(0, len(crops), 16):
            x = np.stack([((resize(c[:, :, ::-1], S, S, cv2.INTER_LINEAR).astype(np.float32) / 255 - m) / s).transpose(2, 0, 1)
                          for c in crops[i:i + 16]]).astype(np.float32)
            lg = self.sid.run(None, {self.sid.get_inputs()[0].name: x})[0]
            out += [self.classes[j] for j in lg.argmax(1)]
        return out

    # ---- 4. recognition
    def recognise(self, lang, crops):
        cfg = self.pcfg[lang]; H, W = cfg["img_size"]; itos = cfg["itos"]; eos = cfg["eos_id"]
        allowed = set(cfg["charset_test"]); lower = cfg["charset_test"] == cfg["charset_test"].lower()
        res = []
        for i in range(0, len(crops), 32):
            x = np.stack([((resize(c[:, :, ::-1], W, H, cv2.INTER_CUBIC).astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)
                          for c in crops[i:i + 32]]).astype(np.float32)
            lg = self.rec[lang].run(None, {self.rec[lang].get_inputs()[0].name: x})[0]
            p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            for row in p:
                ids, pr = row.argmax(-1), row.max(-1)
                n = int(np.argmax(ids == eos)) if (ids == eos).any() else len(ids)
                txt = "".join(itos[t] for t in ids[:n])
                if lower:
                    txt = txt.lower()
                txt = "".join(ch for ch in txt if ch in allowed)
                res.append((txt, float(pr[: n + 1].mean())))
        return res

    def run(self, bgr, si_mode="model"):
        """si_mode 'model': EfficientNetV2-L script ID picks the recogniser.
        si_mode 'dual': no script-ID model; every word is read by both Hindi and English PARSeq and
        the reading with the higher confidence is kept (si = 'hindi' / 'english' accordingly)."""
        if si_mode == "dual":
            return self.run_dual(bgr)
        t = {}; t0 = time.perf_counter()
        boxes, scores = self.detect(bgr); t["detect"] = time.perf_counter() - t0
        crops, keep = [], []
        for j, b in enumerate(boxes):
            c = self.crop(bgr, b)
            if c is not None:
                crops.append(c); keep.append(j)
        t0 = time.perf_counter(); langs = self.script_id(crops) if crops else []; t["script_id"] = time.perf_counter() - t0
        t0 = time.perf_counter(); words = {}
        for lang in ("hindi", "english"):
            js = [q for q, l in enumerate(langs) if APP_LANGS.get(l) == lang]
            for q, (txt, c) in zip(js, self.recognise(lang, [crops[q] for q in js]) if js else []):
                words[q] = (txt, c)
        t["recognise"] = time.perf_counter() - t0
        out = []
        for q, j in enumerate(keep):
            out.append({"box": boxes[j].tolist(), "score": float(scores[j]), "si": langs[q],
                        "text": words.get(q, (None, None))[0], "conf": words.get(q, (None, None))[1]})
        return out, t


    def run_dual(self, bgr):
        t = {}; t0 = time.perf_counter()
        boxes, scores = self.detect(bgr); t["detect"] = time.perf_counter() - t0
        crops, keep = [], []
        for j, b in enumerate(boxes):
            c = self.crop(bgr, b)
            if c is not None:
                crops.append(c); keep.append(j)
        t["script_id"] = 0.0
        t0 = time.perf_counter()
        hi = self.recognise("hindi", crops) if crops else []
        en = self.recognise("english", crops) if crops else []
        t["recognise"] = time.perf_counter() - t0
        out = []
        for q, j in enumerate(keep):
            (th, ch), (te, ce) = hi[q], en[q]
            lang, txt, c = ("hindi", th, ch) if ch >= ce else ("english", te, ce)
            out.append({"box": boxes[j].tolist(), "score": float(scores[j]), "si": lang, "text": txt, "conf": c})
        return out, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(ROOT / "mobile_app/models"))
    ap.add_argument("--suffix", default="", help="model file suffix, e.g. _fp16 or _int8")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--si-mode", choices=("model", "dual"), default="model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import official_metric_e2e as om
    import run_all_end2end_ensembles as base
    gt = json.load(open(om.GT_PATH, encoding="utf-8"))
    keys = om.split_keys(gt, a.split)[: a.limit or None]
    pipe = Pipeline(a.models, a.suffix, a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    preds = {"__status__": {}}; times = []
    for i, k in enumerate(keys, 1):
        res, t = pipe.run(cv2.imread(base.imgpath(gt[k], str(om.IMAGE_ROOT))), a.si_mode)
        preds[k] = {f"polygon_{j}": {"text": r["text"], "si": r["si"], "conf": r["conf"]} for j, r in enumerate(res) if r["text"] is not None}
        preds["__status__"][k] = "success"; times.append(t)
        if i % 50 == 0:
            print(f"{i}/{len(keys)}", flush=True)
    om.atomic_json(out / "predictions.json", preds)
    gt_sub = {k: gt[k] for k in keys}
    table, _ = om.score_system(om.official(), gt_sub, preds)
    summ = {"images": len(keys), "suffix": a.suffix, "threads": a.threads, "si_mode": a.si_mode,
            "mean_seconds": {s: round(float(np.mean([t[s] for t in times])), 3) for s in times[0]},
            "official": {l: {m: round(100 * table[l][m], 2) for m in ("precision", "recall", "f1", "wrr", "crr")} for l in table}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"mean_seconds": summ["mean_seconds"], "average": summ["official"]["average"],
                      "hindi": summ["official"].get("hindi"), "english": summ["official"].get("english")}, indent=1))


if __name__ == "__main__":
    main()
