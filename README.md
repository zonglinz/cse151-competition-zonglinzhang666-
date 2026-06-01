# CSE151B SP26 Competition

This repository contains a single-entry inference pipeline for the CSE151B Spring 2026 math reasoning competition.

## Quick Start

Place `private.jsonl` in the repository root, install dependencies, then run:

```bash
pip install -r requirements.txt
python run_inference.py --private-path private.jsonl --output-csv submission.csv --engine auto --profile v106
```

The command writes a Kaggle-style CSV:

```csv
id,response
```

The same entry point can also be called from Python:

```python
from run_inference import run_inference

run_inference(private_path="private.jsonl", output_csv="submission.csv")
```

## Model And Hardware

Base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

No final fine-tuned checkpoint is required. The model weights download automatically from HuggingFace Hub through vLLM or Transformers.

Main GPU used:

```text
NVIDIA RTX 4090 24GB
```

Approximate full private-set runtime:

```text
24 to 40 hours on one RTX 4090
```

Runtime depends on CUDA/vLLM setup, batching, and cache state.

## Submission Profiles

### v106: default submitted profile

```bash
python run_inference.py --private-path private.jsonl --output-csv submission.csv --engine auto --profile v106
```

This profile uses Qwen generation, answer extraction, self-consistency voting, and final reboxing. It does not use private-id numeric answer replacement.

### v121: adaptive MCQ candidate

```bash
python run_inference.py --private-path private.jsonl --output-csv submission_v121.csv --engine auto --profile v121
```

This profile first runs the v106 pipeline, then reruns multiple-choice rows with an elimination prompt. If the second Qwen-produced option letter differs, that model response is used for that MCQ row.

## Method Summary

Free-form questions:

- Generate multiple Qwen samples.
- Extract each final `\boxed{...}` answer.
- Vote over normalized model-produced answers.
- Keep the winning model trace and append a clean final boxed answer.

Multiple-choice questions:

- Generate multiple Qwen samples.
- Extract boxed option letters.
- Vote over model-produced letters.
- For `profile v121`, run one additional Qwen elimination pass on MCQ rows.

The post-processing is limited to extraction, voting, reboxing, and adaptive model reruns. It does not include a private-id to numeric-answer repair map.

## Hyperparameters

Main pass:

```text
seed = 151
engine = auto
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

v121 MCQ elimination pass:

```text
samples = 1
max_tokens = 26000
prompt style = elimination
```

## Included Files

Core pipeline:

```text
run_inference.py
cse151b_qwen_highscore.py
validate_submission.py
requirements.txt
```

Reference outputs:

```text
submission/submission.csv
submission/submission_v106_jump_hardvote_probe.csv
submission/submission_v121_mcqelim26000_letterchanges_on_v106.csv
```

Optional dependency file:

```text
requirements_vllm.txt
```

Dataset files such as `private.jsonl` are not included in the repository.

## Validation

Validate a produced CSV with:

```bash
python validate_submission.py --input private.jsonl --submission submission.csv
```

For public/development smoke tests, use:

```bash
python run_inference.py --private-path public.jsonl --output-csv public_smoke.csv --engine auto --profile v106
```
