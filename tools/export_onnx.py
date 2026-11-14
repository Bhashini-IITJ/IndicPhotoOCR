#!/usr/bin/env python3
"""Export the demo pipeline to ONNX for in-browser inference (onnxruntime-web).

  yolo.onnx           YOLO11n BSTD + MIST (fork default yolo11n_bstd_mist_1920), dynamic H/W,
                      raw head output (1, 5, N) = cx, cy, w, h, score; NMS is done in JS
  script_id.onnx      EfficientNetV2-L 12-class script identifier, input (B, 3, 288, 288)
  parseq_<lang>.onnx  PARSeq recogniser per script, input (B, 3, H, W) -> logits (B, T, C)
  manifest.json       classes, charsets (itos + adapter charset), image sizes, preprocessing

PARSeq: the forward pass is re-implemented without the data-dependent early stop
(`if all items have EOS: break`) so it can be traced; all max_length+1 steps always run.
Tokens after the first EOS are discarded by the decoder, so outputs are unchanged.
"""
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="IndicPhotoOCR fork checkout with downloaded checkpoints")
ap.add_argument("--out", required=True)
ap.add_argument("--only", nargs="*", default=None)
a = ap.parse_args()
sys.path.insert(0, a.src)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
manifest = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").is_file() else {}


def want(name):
    return a.only is None or name in a.only


# ---------------------------------------------------------------- YOLO
if want("yolo"):
    from ultralytics import YOLO
    from IndicPhotoOCR.detection.yolo import yolo_detector as yd
    info = yd.model_info[yd.DEFAULT_MODEL]
    ckpt = Path(a.src) / "IndicPhotoOCR/detection/yolo" / info["path"] / info["filename"]
    yd.download_model_from_release(yd.DEFAULT_MODEL, ckpt)
    f = YOLO(str(ckpt)).export(format="onnx", imgsz=info["imgsz"], dynamic=True, simplify=True, opset=17, nms=False)
    Path(f).replace(OUT / "yolo.onnx")
    manifest["yolo"] = {"file": "yolo.onnx", "imgsz": info["imgsz"], "conf": 0.25, "iou": 0.7, "max_det": 300,
                        "stride": 32, "letterbox_color": 114, "sha256_pt": info["sha256"]}
    print("yolo exported")

# ---------------------------------------------------------------- script ID
if want("script_id"):
    from IndicPhotoOCR.script_identification.efficientnet.efficientnet_infer import EfficientNetIdentifier
    sid = EfficientNetIdentifier(device="cpu")
    m = sid.model.eval()
    x = torch.randn(1, 3, 288, 288)
    torch.onnx.export(m, x, OUT / "script_id.onnx", input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    manifest["script_id"] = {"file": "script_id.onnx", "classes": sid.classes, "size": 288,
                             "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    print("script_id exported", sid.classes)

# ---------------------------------------------------------------- PARSeq


class ParseqFull(nn.Module):
    """PARSeq.forward (testing mode) without the early break: always max_length + 1 AR steps."""

    def __init__(self, system):
        super().__init__()
        self.m = system.model
        self.tok = system.tokenizer

    def forward(self, images):
        m, tok = self.m, self.tok
        num_steps = m.max_label_length + 1
        bs = images.shape[0]
        memory = m.encode(images)
        pos_queries = m.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool), 1)
        if m.decode_ar:
            tgt_in = torch.full((bs, num_steps), tok.pad_id, dtype=torch.long)
            tgt_in[:, 0] = tok.bos_id
            logits = []
            for i in range(num_steps):
                j = i + 1
                tgt_out = m.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                   tgt_query_mask=query_mask[i:j, :j])
                p_i = m.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i[:, 0].argmax(-1)
            logits = torch.cat(logits, dim=1)
        else:
            tgt_in = torch.full((bs, 1), tok.bos_id, dtype=torch.long)
            logits = m.head(m.decode(tgt_in, memory, tgt_query=pos_queries))
        if m.refine_iters:
            query_mask = query_mask.clone()
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool), 2)] = 0
            bos = torch.full((bs, 1), tok.bos_id, dtype=torch.long)
            for _ in range(m.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == tok.eos_id).int().cumsum(-1) > 0
                tgt_out = m.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]])
                logits = m.head(tgt_out)
        return logits


def load_parseq(lang):
    if lang == "english":
        return torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()
    from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser, model_info
    from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
    path = PARseqrecogniser().ensure_model(lang)
    return load_from_checkpoint(path).eval()


from IndicPhotoOCR.recognition.parseq_recogniser import model_info as rec_info
langs = sorted(rec_info) + ["english"]
manifest.setdefault("parseq", {})
for lang in langs:
    if not want(f"parseq_{lang}") and not want("parseq"):
        continue
    sysm = load_parseq(lang)
    hp = sysm.hparams
    h, w = hp.img_size
    net = ParseqFull(sysm).eval()
    x = torch.rand(2, 3, h, w) * 2 - 1
    ref = sysm(x)  # original forward (may stop early)
    got = net(x)
    T = ref.shape[1]
    diff = (ref.softmax(-1) - got[:, :T].softmax(-1)).abs().max().item()
    f = OUT / f"parseq_{lang}.onnx"
    torch.onnx.export(net, x, f, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    itos = list(sysm.tokenizer._itos)
    manifest["parseq"][lang] = {"file": f.name, "img_size": [h, w], "itos": itos, "eos_id": sysm.tokenizer.eos_id,
                                "charset_test": hp.charset_test, "max_label_length": int(sysm.model.max_label_length),
                                "refine_iters": int(sysm.model.refine_iters), "decode_ar": bool(sysm.model.decode_ar)}
    print(f"parseq {lang}: img {h}x{w}, {len(itos)} tokens, max_len {sysm.model.max_label_length}, "
          f"refine {sysm.model.refine_iters}, prob diff vs original {diff:.2e}", flush=True)

(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
