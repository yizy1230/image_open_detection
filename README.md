# image_open_detection v1_0

Open-vocabulary detection with **LLMDet** (`checkpoints/LLMDet/tiny.pth`) and the vocabulary in `open_free_vocab.py`. By default it scans **core** subjects and environment elements and returns boxes, scores, and roles. Model code is in `third_party/LLMDet`. Weight download instructions are in `checkpoints/WEIGHTS.txt`.

```bash
export LLMDET_ROOT=third_party/LLMDet
export LLMDET_MODEL_DIR=checkpoints/LLMDet
PYTHONPATH=.:$LLMDET_ROOT python image_open_detection.py \
  --input-json test/test_local_1.json --output-json test/result.json --device cuda
```

Build the worker from this repository root:

```bash
docker build -f Dockerfile -t algo-image_open_detection:v1_0 .
```

Worker contract: `process_image(path, ModelManager, ...)`. The full JSON is stored in MinIO. `tasks.output` keeps only counts and a label summary.

## Weights

Weight files are not stored in this repository. Download URLs and destination paths are in `checkpoints/WEIGHTS.txt`.

## Citation

```bibtex
@software{image_open_detection,
  title   = {image_open_detection},
  year    = {2026},
  author  = {yizy},
  license = {Apache-2.0}
}
```

## License

Code written for this repository is under the Apache License 2.0. See [LICENSE](LICENSE). Third-party directories keep their own licenses. See [NOTICE](NOTICE).

## Release

- **v1.0.0** — First public release. Open-vocabulary detection with the LLMDet tiny checkpoint and this repository's vocabulary, plus a Celery worker.

## Future Work

- Switch the LLMDet variant and the vocabulary file without editing code.
