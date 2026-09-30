#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open-vocabulary detection with LLMDet + shared subject/environment vocab.

Weights are the mmdet checkpoints under ``checkpoints/LLMDet`` (tiny.pth by
default). See ``checkpoints/WEIGHTS.txt``. Inference uses LLMDet's Grounding
DINO with ``lmm=None``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.instance_codec import box_area, class_aware_nms  # noqa: E402
from open_free_vocab import get_entry, get_vocab  # noqa: E402

DEFAULT_LLMDET_ROOT = Path(
    os.environ.get("LLMDET_ROOT", str(SCRIPT_DIR / "third_party" / "LLMDet"))
)
DEFAULT_MODEL_ZOO = Path(
    os.environ.get("LLMDET_MODEL_DIR", str(SCRIPT_DIR / "checkpoints" / "LLMDet"))
)
WEIGHT_FILES = {
    "tiny": "tiny.pth",
    "tiny_p5": "tiny_p5.pth",
    "base": "base.pth",
    "large": "large.pth",
}
CONFIG_FILES = {
    "tiny": "configs/grounding_dino_t_for_alignment.py",
    "tiny_p5": "configs/grounding_dino_swin_t_only_p5.py",
    "base": "configs/grounding_dino_swin_b.py",
    "large": "configs/grounding_dino_swin_l.py",
}
DEFAULT_VARIANT = "tiny"
DEFAULT_SCORE_THR = 0.35
DEFAULT_CHUNK_SIZE = 40
DEFAULT_MAX_SIDE = 1920
DEFAULT_NMS_IOU = 0.5


def _resolve_weights(variant: str, weights: Optional[str] = None) -> Path:
    if weights:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"LLMDet weights not found: {path}")
        return path
    name = WEIGHT_FILES.get(variant, WEIGHT_FILES[DEFAULT_VARIANT])
    path = DEFAULT_MODEL_ZOO / name
    if not path.is_file():
        raise FileNotFoundError(
            f"LLMDet weights not found: {path}. Set LLMDET_WEIGHTS or LLMDET_MODEL_DIR."
        )
    return path


def _resolve_names(
    names: Optional[Sequence[str]],
    tier: str,
    role: Optional[str],
) -> List[str]:
    if names:
        return [str(n).strip() for n in names if str(n).strip()]
    return get_vocab(model="llmdet", role=role, tier=tier, lang="en")


def _annotate(label: str, score: float, box: Sequence[float]) -> Dict[str, Any]:
    meta = get_entry(label) or {}
    return {
        "label": meta.get("en", label),
        "label_zh": meta.get("zh", ""),
        "role": meta.get("role", "subject"),
        "score": round(float(score), 4),
        "bbox_xyxy": [round(float(x), 2) for x in box[:4]],
        "area": round(box_area(box), 1),
    }


def pack_detections(
    labels: Sequence[str],
    scores: Sequence[float],
    boxes: Sequence[Sequence[float]],
    score_thr: float = DEFAULT_SCORE_THR,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_per_label: int = 20,
    max_total: int = 200,
) -> List[Dict[str, Any]]:
    dets = []
    for lab, score, box in zip(labels, scores, boxes):
        if float(score) < score_thr:
            continue
        dets.append(_annotate(str(lab), float(score), box))
    if not dets:
        return []
    keep = class_aware_nms(
        [d["bbox_xyxy"] for d in dets],
        [d["score"] for d in dets],
        [d["label"] for d in dets],
        iou_thr=nms_iou,
    )
    dets = [dets[i] for i in keep]
    per_label: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []
    for det in dets:
        n = per_label.get(det["label"], 0)
        if n >= max_per_label:
            continue
        per_label[det["label"]] = n + 1
        filtered.append(det)
        if len(filtered) >= max_total:
            break
    return filtered


def summarize(dets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    subjects: List[str] = []
    environment: List[str] = []
    for det in dets:
        counts[det["label"]] = counts.get(det["label"], 0) + 1
        bucket = subjects if det.get("role") == "subject" else environment
        if det["label"] not in bucket:
            bucket.append(det["label"])
    return {
        "num_instances": len(dets),
        "label_summary": counts,
        "subjects": subjects,
        "environment": environment,
        "num_subjects": sum(1 for d in dets if d.get("role") == "subject"),
        "num_environment": sum(1 for d in dets if d.get("role") == "environment"),
    }


def draw_detections(image: Image.Image, dets: Sequence[Dict[str, Any]]) -> Image.Image:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    colors = {"subject": (255, 64, 64), "environment": (64, 160, 255), "part": (255, 200, 64)}
    for det in dets:
        color = colors.get(det.get("role"), (0, 200, 80))
        x0, y0, x1, y1 = det["bbox_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        caption = f"{det['label']} {det['score']:.2f}"
        draw.text((x0 + 2, max(0, y0 - 12)), caption, fill=color, font=font)
    return vis


class ModelManager:
    def __init__(
        self,
        device: str = "cuda",
        variant: str = DEFAULT_VARIANT,
        weights: Optional[str] = None,
        llmdet_root: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        bert_name: str = "bert-base-uncased",
    ):
        self.device = device
        self.variant = variant
        self.weights = str(_resolve_weights(variant, weights or os.environ.get("LLMDET_WEIGHTS")))
        self.llmdet_root = Path(llmdet_root or DEFAULT_LLMDET_ROOT)
        self.chunk_size = chunk_size
        self.bert_name = os.environ.get("BERT_NAME", bert_name)
        self._inferencer = None

    def get_inferencer(self):
        if self._inferencer is None:
            self._inferencer = _build_inferencer(
                self.llmdet_root,
                self.weights,
                self.variant,
                self.device,
                self.chunk_size,
                self.bert_name,
            )
        return self._inferencer


def _build_inferencer(
    llmdet_root: Path,
    weights: str,
    variant: str,
    device: str,
    chunk_size: int,
    bert_name: str,
):
    if not llmdet_root.is_dir():
        raise FileNotFoundError(f"LLMDet project not found: {llmdet_root}. Set LLMDET_ROOT.")
    root = str(llmdet_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config
    from mmdet.apis import DetInferencer

    rel = CONFIG_FILES.get(variant, CONFIG_FILES[DEFAULT_VARIANT])
    cfg_path = llmdet_root / rel
    if not cfg_path.is_file():
        raise FileNotFoundError(f"LLMDet config not found: {cfg_path}")
    cwd = os.getcwd()
    try:
        os.chdir(root)
        cfg = Config.fromfile(str(cfg_path))
        cfg.model.lmm = None
        if hasattr(cfg.model, "language_model") and cfg.model.language_model is not None:
            cfg.model.language_model.name = bert_name
        tmp = tempfile.NamedTemporaryFile(prefix="llmdet_infer_", suffix=".py", delete=False)
        cfg.dump(tmp.name)
        inferencer = DetInferencer(model=tmp.name, weights=weights, device=device, palette="none")
    finally:
        os.chdir(cwd)
    if hasattr(inferencer.model, "test_cfg") and inferencer.model.test_cfg is not None:
        inferencer.model.test_cfg.chunked_size = chunk_size
    return inferencer


def _resize_max_side(image: Image.Image, max_side: Optional[int]) -> Image.Image:
    if not max_side or max_side <= 0:
        return image
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)


def _labels_from_prediction(pred: Dict[str, Any], names: Sequence[str]) -> List[str]:
    if pred.get("label_names"):
        return [str(x) for x in pred["label_names"]]
    out = []
    for idx in pred.get("labels") or []:
        i = int(idx)
        out.append(names[i] if 0 <= i < len(names) else str(idx))
    return out


def process_image(
    image_path: str,
    model_mgr: ModelManager,
    names: Optional[Sequence[str]] = None,
    vocab_tier: str = "core",
    vocab_role: Optional[str] = None,
    score_thr: float = DEFAULT_SCORE_THR,
    nms_iou: float = DEFAULT_NMS_IOU,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_side: int = DEFAULT_MAX_SIDE,
    max_per_label: int = 20,
    max_total: int = 200,
    return_vis: bool = False,
    **_kwargs,
) -> Dict[str, Any]:
    t0 = time.time()
    timings: Dict[str, float] = {}
    names = _resolve_names(names, vocab_tier, vocab_role)
    if not names:
        raise ValueError("empty open-vocab list")

    t = time.time()
    image = Image.open(image_path).convert("RGB")
    image = _resize_max_side(image, max_side)
    tmp_path = None
    infer_path = image_path
    if max_side and max(image.size) != max(Image.open(image_path).size):
        tmp = tempfile.NamedTemporaryFile(prefix="llmdet_", suffix=".jpg", delete=False)
        image.save(tmp.name, quality=95)
        tmp_path = tmp.name
        infer_path = tmp_path
    timings["load_image"] = round(time.time() - t, 3)

    inferencer = model_mgr.get_inferencer()
    if hasattr(inferencer.model, "test_cfg") and inferencer.model.test_cfg is not None:
        inferencer.model.test_cfg.chunked_size = chunk_size

    t = time.time()
    raw = inferencer(
        infer_path,
        texts=[tuple(names)],
        custom_entities=True,
        pred_score_thr=score_thr,
        no_save_vis=True,
        no_save_pred=True,
        return_vis=False,
        show=False,
        out_dir="",
    )
    timings["infer"] = round(time.time() - t, 3)
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    preds = (raw or {}).get("predictions") or [{}]
    pred = preds[0] if preds else {}
    labels = _labels_from_prediction(pred, names)
    scores = pred.get("scores") or []
    boxes = pred.get("bboxes") or []
    dets = pack_detections(
        labels, scores, boxes,
        score_thr=score_thr,
        nms_iou=nms_iou,
        max_per_label=max_per_label,
        max_total=max_total,
    )
    summary = summarize(dets)
    timings["total"] = round(time.time() - t0, 3)

    result: Dict[str, Any] = {
        "image_path": image_path,
        "width": image.size[0],
        "height": image.size[1],
        "vocab_tier": vocab_tier,
        "vocab_size": len(names),
        "score_thr": score_thr,
        "detections": dets,
        "timings": timings,
        "algo_name": "image_open_detection",
        "algo_version": "v1_0",
        "model": {
            "name": "LLMDet",
            "variant": model_mgr.variant,
            "weights": model_mgr.weights,
        },
    }
    result.update(summary)
    if return_vis:
        result["vis_image"] = draw_detections(image, dets)
    return result


def main():
    parser = argparse.ArgumentParser(description="image_open_detection v1.0 (LLMDet)")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(WEIGHT_FILES))
    parser.add_argument("--vis", default="", help="optional overlay PNG path")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)
    image_path = inp.get("image_path") or inp.get("video_path")
    if not image_path:
        raise ValueError("image_path is required")

    mgr = ModelManager(device=args.device, variant=args.variant)
    result = process_image(
        image_path,
        mgr,
        names=inp.get("names"),
        vocab_tier=inp.get("vocab_tier", "core"),
        vocab_role=inp.get("vocab_role"),
        score_thr=float(inp.get("score_thr", DEFAULT_SCORE_THR)),
        return_vis=bool(args.vis),
    )
    vis = result.pop("vis_image", None)
    if args.vis and vis is not None:
        Path(args.vis).parent.mkdir(parents=True, exist_ok=True)
        vis.save(args.vis)
        result["vis_path"] = args.vis
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"instances={result['num_instances']} subjects={result['subjects']} "
        f"env={result['environment']} -> {args.output_json}"
    )


if __name__ == "__main__":
    main()
