#!/usr/bin/env python3
"""Prune overlapping detections of the same text, using only the boxes and the text read in them
(no lexicon, no training data). The app (OcrEngine.prune) implements exactly this rule.

1. Same text twice: two boxes covering the same place (overlap >= 0.5 of the smaller box) with the
   same text -> keep the higher detector score.
2. Nested boxes: boxes lying inside a bigger box B (>= 0.7 of their area). Among them, the ones that
   do not overlap each other are the "parts" of B. Bigger boxes are handled first.
   a) English B (phrase boxes): 2+ parts spanning >= 75% of B's width -> B is a phrase box over
      separate words: drop B if every part beats it on reader confidence OR on detector score
      (phrase boxes like "waterpurifier" get a much lower detector score than "water", "purifier";
      a misread phrase like "leareenly" gets a lower reader confidence than "leave", "only").
   b) Otherwise only inner boxes whose text is part of B's text count: 2+ parts covering >= 60% of
      B's text -> drop B if the reader is more confident in every part; else the inner boxes are
      fragments of B, each dropped if the reader is at least as confident in B.
   Rule (a) is English only: in Indic scripts the detector often splits one word into touching
   pieces and the bigger box is the right one (val, all scripts: Bengali recall -2.7).
   Boxes that do not overlap are never touched, so the same word written in two places is kept.
Text is compared lower-cased and without spaces.

Example (validation only):
  python dedup_boxes.py --pred ../eval/val_all_int8_boxes/predictions.json --out ../eval/val_all_int8_dedup
"""
import argparse, json, sys
from pathlib import Path

SAME_OVERLAP, INSIDE, PART_OVERLAP, COVER, SPAN = 0.5, 0.7, 0.3, 0.6, 0.75


def norm(t):
    return "".join((t or "").lower().split())


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def prune(words):
    """words: list of dicts with box [x1,y1,x2,y2], score, text. Returns the indices to drop, with reasons."""
    n = len(words); txt = [norm(w["text"]) for w in words]; ar = [area(w["box"]) for w in words]
    cf = [w.get("conf") or 0.0 for w in words]
    drop = {}
    # 1. same text twice
    for i in range(n):
        for j in range(i + 1, n):
            if i in drop or j in drop or not txt[i] or txt[i] != txt[j]:
                continue
            m = min(ar[i], ar[j])
            if m > 0 and inter(words[i]["box"], words[j]["box"]) / m >= SAME_OVERLAP:
                k = j if words[j]["score"] < words[i]["score"] else i
                drop[k] = "duplicate"
    # 2. nested boxes, biggest first
    sc = [w["score"] for w in words]; B = [w["box"] for w in words]
    for b in sorted(range(n), key=lambda q: -ar[q]):
        if b in drop or not txt[b]:
            continue
        inside = [i for i in range(n) if i != b and i not in drop and txt[i] and 0 < ar[i] < ar[b]
                  and inter(B[i], B[b]) / ar[i] >= INSIDE]
        inner = [i for i in inside if txt[i] != txt[b] and txt[i] in txt[b]]
        phrase = words[b].get("si") == "english"
        parts = []
        for i in sorted(inside if phrase else inner, key=lambda q: -ar[q]):
            if all(inter(B[i], B[p]) / min(ar[i], ar[p]) < PART_OVERLAP for p in parts):
                parts.append(i)
        if phrase:
            joined = len(parts) >= 2 and (max(B[p][2] for p in parts) - min(B[p][0] for p in parts)) / max(1e-6, B[b][2] - B[b][0]) >= SPAN
            if joined:
                if min(cf[p] for p in parts) > cf[b] or min(sc[p] for p in parts) > sc[b]:
                    drop[b] = "merged"
                continue
        elif len(parts) >= 2 and sum(len(txt[p]) for p in parts) >= COVER * len(txt[b]):
            if min(cf[p] for p in parts) > cf[b]:
                drop[b] = "merged"
            continue
        for i in inner:
            if cf[b] >= cf[i]:
                drop[i] = "fragment"
    return drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, nargs="+", help="predictions.json with box and score per word (several = shards, merged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test", action="store_true", help="allow BSTD test images (final one-time check)")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    import official_metric_e2e as om
    gt = json.load(open(om.GT_PATH, encoding="utf-8"))
    pred = {"__status__": {}}
    for f in a.pred:
        p = json.load(open(f, encoding="utf-8"))
        pred["__status__"].update(p.pop("__status__")); pred.update(p)
    keys = [k for k in pred if k != "__status__"]
    if any(gt[k].get("split") != "train" for k in keys) and not a.allow_test:
        sys.exit("validation (train-split) images only unless --allow-test")
    out = {"__status__": pred["__status__"]}; reasons = {}
    for k in keys:
        ids = list(pred[k]); words = [pred[k][i] for i in ids]
        d = prune(words)
        for r in d.values():
            reasons[r] = reasons.get(r, 0) + 1
        out[k] = {ids[q]: words[q] for q in range(len(ids)) if q not in d}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    om.atomic_json(o / "predictions.json", out)
    om.atomic_json(o / "predictions_unpruned.json", pred)
    sub = {k: gt[k] for k in keys}
    res = {}
    for name, p in (("before", pred), ("after", out)):
        table, _ = om.score_system(om.official(), sub, p)
        res[name] = {l: {m: round(100 * table[l][m], 2) for m in ("precision", "recall", "f1", "wrr", "crr")} for l in table}
    n_before = sum(len(pred[k]) for k in keys)
    summ = {"images": len(keys), "words_before": n_before, "dropped": reasons, "official": res}
    (o / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"words_before": n_before, "dropped": reasons}))
    for l in res["before"]:
        b, f = res["before"][l], res["after"][l]
        print(f"{l:10s} " + "  ".join(f"{m} {b[m]:5.1f}->{f[m]:5.1f} ({f[m] - b[m]:+.2f})" for m in ("precision", "recall", "f1", "wrr", "crr")))


if __name__ == "__main__":
    main()
