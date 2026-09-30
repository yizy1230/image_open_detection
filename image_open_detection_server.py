"""FastAPI server for image_open_detection."""

import argparse
import os
import sys

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_open_detection import ModelManager, process_image

app = FastAPI(title="image_open_detection", version="v1_0")
_mgr = None


def _mgr_get(device: str) -> ModelManager:
    global _mgr
    if _mgr is None:
        _mgr = ModelManager(device=device)
    return _mgr


class DetectRequest(BaseModel):
    image_path: str
    device: str = "cuda"
    names: Optional[List[str]] = None
    vocab_tier: str = "core"
    vocab_role: Optional[str] = None
    score_thr: float = 0.35
    chunk_size: int = 40


@app.post("/image_open_detection")
async def detect(req: DetectRequest):
    try:
        result = process_image(
            req.image_path,
            _mgr_get(req.device),
            names=req.names,
            vocab_tier=req.vocab_tier,
            vocab_role=req.vocab_role,
            score_thr=req.score_thr,
            chunk_size=req.chunk_size,
        )
        result.pop("vis_image", None)
        return result
    except Exception as exc:
        logger.exception("open detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ready")
async def ready():
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
