#!/usr/bin/env python3
"""Single-entry CSE151B inference pipeline.

Calling run_inference() performs the full private-set pipeline:
1. Generate raw Qwen responses with cse151b_qwen_highscore.py.
2. Vote/extract/rebox the model-produced answers.
3. Write and validate the Kaggle CSV with columns: id,response.

The default profile is "v106" so final answer values come from model inference
rather than private-ID numeric replacement. Profile "v121" adds an adaptive
MCQ elimination second pass, also model-inference based.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping

import cse151b_qwen_highscore as qwen


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


PROFILE_CHOICES = ("none", "v106", "v121")


def write_final_submission(base_csv: str | Path, output_csv: str | Path, replacements: Mapping[int, str] | None = None) -> None:
    """Write final CSV after extractor-safe model-output formatting."""
    replacements = replacements or {}
    base_csv = Path(base_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with base_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "response"]:
            raise ValueError(f"{base_csv} must have header id,response")
        rows = list(reader)

    for row in rows:
        rid = int(row["id"])
        if rid in replacements:
            row["response"] = replacements[rid]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(rows)


def _generation_args(
    private_path: str | Path,
    base_csv: str | Path,
    checkpoint_path: str | Path,
    model_id: str,
    engine: str,
    repo_dir: str | Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        input=str(private_path),
        output=str(base_csv),
        checkpoint=str(checkpoint_path),
        repo_dir=str(repo_dir),
        model=model_id,
        engine=engine,
        placeholder_only=False,
        write_from_checkpoint_only=False,
        limit=0,
        seed=151,
        batch_size=8,
        samples=None,
        samples_simple=4,
        samples_hard=12,
        max_tokens=32768,
        max_model_len=131072,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        dtype="bfloat16",
        quantization="none",
        load_format="none",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_num_seqs=64,
        max_num_batched_tokens=65536,
        enable_prefix_caching=False,
        transformers_batch_size=1,
        attn_implementation="sdpa",
        force_final_from_partial=True,
        force_final_max_tokens=256,
        selector_on_tie=False,
        selector_max_tokens=8192,
        selector_temperature=0.2,
        score_public=False,
    )


def _read_csv(path: str | Path) -> Dict[int, str]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {int(r["id"]): r["response"] for r in csv.DictReader(f)}


def _write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_v121_mcq_elimination_profile(
    private_path: str | Path,
    base_csv: str | Path,
    output_csv: str | Path,
    *,
    model_id: str,
    engine: str,
    repo_dir: str | Path,
    checkpoint_path: str | Path,
) -> None:
    """Adaptive MCQ second pass for the v121-style candidate.

    This does not use private id -> answer constants. It reruns Qwen on MCQ
    rows with an elimination prompt and replaces only rows where the
    model-produced final option letter differs from the base pass.
    """
    rows = qwen.load_jsonl(private_path)
    mcq_rows = [r for r in rows if r.get("options")]
    if not mcq_rows:
        write_final_submission(base_csv, output_csv)
        return

    output_csv = Path(output_csv)
    mcq_jsonl = output_csv.with_suffix(".mcq_elim_input.jsonl")
    mcq_csv = output_csv.with_suffix(".mcq_elim.csv")
    mcq_checkpoint = Path(str(checkpoint_path) + ".mcq_elim.jsonl")
    _write_jsonl(mcq_jsonl, mcq_rows)

    args = _generation_args(mcq_jsonl, mcq_csv, mcq_checkpoint, model_id, engine, repo_dir)
    args.samples = 1
    args.samples_simple = 1
    args.samples_hard = 1
    args.max_tokens = 26000
    args.batch_size = 8

    old_style = os.environ.get("CSE151B_MCQ_PROMPT_STYLE")
    os.environ["CSE151B_MCQ_PROMPT_STYLE"] = "elimination"
    try:
        qwen.generate(args)
    finally:
        if old_style is None:
            os.environ.pop("CSE151B_MCQ_PROMPT_STYLE", None)
        else:
            os.environ["CSE151B_MCQ_PROMPT_STYLE"] = old_style

    base = _read_csv(base_csv)
    elim = _read_csv(mcq_csv)
    replacements: Dict[int, str] = {}
    for rid, response in elim.items():
        if qwen.extract_letter(base.get(rid, "")) != qwen.extract_letter(response):
            replacements[rid] = response
    write_final_submission(base_csv, output_csv, replacements)


def run_inference(
    private_path: str | Path = "private.jsonl",
    output_csv: str | Path = "submission.csv",
    *,
    model_id: str = MODEL_ID,
    engine: str = "auto",
    repo_dir: str | Path = "151B_SP26_Competition-main",
    checkpoint_path: str | Path = "submission_checkpoint.jsonl",
    profile: str = "v106",
    base_csv: str | Path | None = None,
    skip_generation: bool = False,
) -> str:
    """Run full inference and return the final CSV path.

    Parameters
    ----------
    private_path:
        Path to private.jsonl.
    output_csv:
        Final Kaggle CSV to write.
    model_id:
        HuggingFace model id or local model directory.
    engine:
        "auto", "vllm", or "transformers".
    checkpoint_path:
        JSONL checkpoint for resumable Qwen generation.
    profile:
        "v106" or "none" for the main model-inference/voting pipeline.
        "v121" adds an adaptive Qwen MCQ elimination second pass.
    base_csv:
        Optional intermediate raw-Qwen CSV path. Defaults to output stem + ".base.csv".
    skip_generation:
        If True, reuse base_csv and only run final CSV validation/copy logic.
        This is useful for resume/debug; normal verification should leave it False.
    """
    if profile not in PROFILE_CHOICES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILE_CHOICES}")

    private_path = Path(private_path)
    output_csv = Path(output_csv)
    if base_csv is None:
        base_csv = output_csv.with_suffix(".base.csv")
    base_csv = Path(base_csv)

    if not skip_generation:
        args = _generation_args(private_path, base_csv, checkpoint_path, model_id, engine, repo_dir)
        qwen.generate(args)
    elif not base_csv.exists():
        raise FileNotFoundError(f"skip_generation=True but {base_csv} does not exist")

    if profile == "v121" and not skip_generation:
        apply_v121_mcq_elimination_profile(
            private_path,
            base_csv,
            output_csv,
            model_id=model_id,
            engine=engine,
            repo_dir=repo_dir,
            checkpoint_path=checkpoint_path,
        )
    else:
        write_final_submission(base_csv, output_csv)
    ok, errors = qwen.validate_submission(private_path, output_csv)
    if not ok:
        raise RuntimeError("invalid submission: " + "; ".join(errors[:10]))
    return str(output_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-path", default="private.jsonl")
    parser.add_argument("--output-csv", default="submission.csv")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--engine", choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--repo-dir", default="151B_SP26_Competition-main")
    parser.add_argument("--checkpoint-path", default="submission_checkpoint.jsonl")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="v106")
    parser.add_argument("--base-csv", default=None)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    out = run_inference(
        private_path=args.private_path,
        output_csv=args.output_csv,
        model_id=args.model_id,
        engine=args.engine,
        repo_dir=args.repo_dir,
        checkpoint_path=args.checkpoint_path,
        profile=args.profile,
        base_csv=args.base_csv,
        skip_generation=args.skip_generation,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
