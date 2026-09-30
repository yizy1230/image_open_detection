#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Celery worker for image_open_detection (LLMDet)."""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from celery.signals import worker_process_init
from loguru import logger

SCRIPT_DIR = Path(__file__).parent.resolve()
ALGOS_ROOT = SCRIPT_DIR.parent.parent.resolve()
if str(ALGOS_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGOS_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("WORKER_CONFIG_PATH", str(SCRIPT_DIR / "worker_config.yaml"))

from common.worker_base import (  # noqa: E402
    _default_postprocess,
    create_worker_app,
    execute_algo_task,
)
from image_open_detection import ModelManager, process_image  # noqa: E402

app = create_worker_app("image_open_detection")
_model_mgr = None
_DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda")


def _get_model_mgr(device: str = "cuda") -> ModelManager:
    global _model_mgr
    if _model_mgr is None:
        _model_mgr = ModelManager(device=device)
    return _model_mgr


@worker_process_init.connect
def _preload_and_warmup(**kwargs):
    global _model_mgr
    t0 = time.time()
    logger.info("[image_open_detection] preloading LLMDet on device={}", _DEFAULT_DEVICE)
    _model_mgr = ModelManager(device=_DEFAULT_DEVICE)
    _model_mgr.get_inferencer()
    logger.info("[image_open_detection] preload done in {:.2f}s", time.time() - t0)


def run_fn(local_input_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    device = params.pop("device", _DEFAULT_DEVICE)
    params.pop("job_id", None)
    params.pop("object_key", None)
    mgr = _get_model_mgr(device)
    result = process_image(local_input_path, mgr, **params)
    result.pop("vis_image", None)
    return result


def postprocess_fn(ctx: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    post = _default_postprocess(ctx, result)
    for key in (
        "num_instances",
        "num_subjects",
        "num_environment",
        "subjects",
        "environment",
        "label_summary",
    ):
        if key in result:
            post["output"][key] = result[key]
    return post


@app.task(name="algo.image_open_detection", bind=True, max_retries=1, default_retry_delay=15, acks_late=True)
def run_task(self, task_id: str) -> Dict[str, Any]:
    return execute_algo_task(
        task_id, celery_task_id=self.request.id, run_fn=run_fn, postprocess_fn=postprocess_fn,
    )
