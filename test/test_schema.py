#!/usr/bin/env python3
"""Schema tests that do not load LLMDet weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.instance_codec import class_aware_nms, mask_to_rle, rle_to_mask
from open_free_vocab import get_vocab
from image_open_detection import pack_detections, summarize
import numpy as np


def test_vocab_llmdet_core():
    names = get_vocab(model="llmdet", tier="core")
    assert "person" in names
    assert "sky" in names
    assert "face" not in names


def test_pack_and_nms():
    dets = pack_detections(
        labels=["person", "person", "car"],
        scores=[0.9, 0.8, 0.7],
        boxes=[[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 80, 80]],
        score_thr=0.3,
        nms_iou=0.5,
    )
    labels = [d["label"] for d in dets]
    assert labels.count("person") == 1
    assert "car" in labels
    summary = summarize(dets)
    assert summary["num_instances"] == 2
    assert "person" in summary["subjects"]


def test_rle_roundtrip():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    back = rle_to_mask(mask_to_rle(mask))
    assert np.array_equal(mask, back)


if __name__ == "__main__":
    test_vocab_llmdet_core()
    test_pack_and_nms()
    test_rle_roundtrip()
    print("ok")
