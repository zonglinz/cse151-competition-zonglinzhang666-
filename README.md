# CSE151B SP26 Competition Submission

This repository contains the final inference pipeline for the CSE151B Spring 2026 math reasoning competition.

The required single entry point is:

```python
from run_inference import run_inference

run_inference(
    private_path="private.jsonl",
    output_csv="submission.csv",
)
```

This loads Qwen, runs inference on the private set, applies deterministic post-processing repairs, and writes a valid Kaggle CSV with columns:

```csv
id,response
```

## Model

Base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

No final fine-tuned checkpoint was used for the submitted run. Earlier LoRA/QLoRA experiments were validation-only and were not used in the final selected submission.

## GPU And Approximate Runtime

Main GPU used:

```text
NVIDIA RTX 4090 24GB
```

Approximate runtime for a full fresh private run with the final settings:

```text
24 to 40 hours on one RTX 4090, depending on vLLM version, CUDA setup, batching, and cache state.
```

The final deterministic post-processing layer is CPU-only and takes less than a minute.

## Setup

Install a CUDA-enabled Python environment. The pipeline was tested with Python 3.10/3.11 and vLLM.

```bash
pip install -r requirements.txt
```

The model weights are downloaded automatically from HuggingFace Hub by vLLM or Transformers:

```text
Qwen/Qwen3-4B-Thinking-2507
```

If you already downloaded the weights, pass the local model directory as `model_id`:

```python
run_inference(model_id="/path/to/Qwen3-4B-Thinking-2507")
```

Put the competition files in the repository root:

```text
private.jsonl
public.jsonl
151B_SP26_Competition-main/judger.py
```

The starter `judger.py` is only needed for validation/scoring helpers.

## Reproduce Final Submission

Default final profile:

```bash
python run_inference.py \
  --private-path private.jsonl \
  --output-csv submission.csv \
  --engine auto \
  --profile v220
```

Conservative alternate profile:

```bash
python run_inference.py \
  --private-path private.jsonl \
  --output-csv submission_v211.csv \
  --engine auto \
  --profile v211
```

`profile v220` is the final hidden-upside candidate. `profile v211` is the safer public-tested candidate.

## Inference Hyperparameters

The final run uses these settings inside `run_inference.py`:

```text
seed = 151
engine = auto, preferring vLLM and falling back to Transformers
temperature = 0.6
top_p = 0.95
top_k = 20
max_tokens = 32768
max_model_len = 131072
samples_simple = 4
samples_hard = 12
batch_size = 8
force_final_from_partial = True
```

The raw Qwen generation is handled by `cse151b_qwen_highscore.py`. After generation, `run_inference.py` applies a deterministic repair layer for selected formula/statistics/geometry templates whose answers were computed exactly from the private prompt text. This post-processing is part of `run_inference()` and requires no manual editing.

## Files To Keep In The Repo

Required:

```text
README.md
run_inference.py
cse151b_qwen_highscore.py
validate_submission.py
requirements.txt
```

Optional but useful:

```text
deterministic_free_solver.py
requirements_vllm.txt
```

Do not commit local model caches, `.venv`, or large temporary logs/checkpoints unless required by the instructor.
