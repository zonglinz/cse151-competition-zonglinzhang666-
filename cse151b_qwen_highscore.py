#!/usr/bin/env python3
"""
CSE 151B Spring 2026 Competition submission generator.

Rule-compliant goal:
  Generate private.jsonl responses with Qwen/Qwen3-4B-Thinking-2507 only.

What this script does:
  - Reads public.jsonl or private.jsonl JSONL problem files.
  - Builds extractor-aligned prompts for MCQ and free-form math.
  - Runs Qwen/Qwen3-4B-Thinking-2507 using vLLM or Transformers.
  - Uses same-model self-consistency: multiple Qwen samples per problem.
  - Extracts final \boxed{...} answers using logic aligned with starter judger.py.
  - Votes by normalized final answer and keeps the winning full raw model trace.
  - Appends a canonical final \boxed{...} line to reduce extraction failures.
  - Writes Kaggle CSV: id,response with proper CSV quoting.
  - Checkpoints every problem so interrupted long runs can resume.
  - Can evaluate public.jsonl locally using the starter judger.

Recommended private run:
  python cse151b_qwen_highscore.py \
    --input private.jsonl \
    --output submission.csv \
    --repo-dir 151B_SP26_Competition-main \
    --engine vllm \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --max-model-len 131072 \
    --max-tokens 32768 \
    --samples-simple 4 \
    --samples-hard 12 \
    --batch-size 8 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20

Higher-budget private run, if you have enough GPU time/memory:
  python cse151b_qwen_highscore.py \
    --input private.jsonl \
    --output submission_high_budget.csv \
    --repo-dir 151B_SP26_Competition-main \
    --engine vllm \
    --max-model-len 262144 \
    --max-tokens 81920 \
    --samples-simple 8 \
    --samples-hard 24 \
    --selector-on-tie \
    --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# JSONL / CSV helpers
# -----------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {e}") from e
    return rows


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint(path: str | Path) -> Dict[int, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["id"])] = row
    return out


def write_submission_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write standards-compliant CSV with id,response columns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(r["id"])):
            writer.writerow({"id": int(row["id"]), "response": str(row["response"])})


def validate_submission(input_jsonl: str | Path, csv_path: str | Path) -> Tuple[bool, List[str]]:
    """Validate Kaggle CSV shape and id coverage."""
    problems = load_jsonl(input_jsonl)
    ids = [int(x["id"]) for x in problems]
    expected = set(ids)
    errors: List[str] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "response"]:
            errors.append(f"CSV header must be exactly ['id', 'response']; got {reader.fieldnames}")
            return False, errors
        rows = list(reader)

    got_ids: List[int] = []
    for i, row in enumerate(rows, 2):
        try:
            rid = int(row["id"])
        except Exception:
            errors.append(f"Row {i}: id is not an integer: {row.get('id')!r}")
            continue
        got_ids.append(rid)
        response = row.get("response", "")
        if not isinstance(response, str) or not response.strip():
            errors.append(f"Row {i}: empty response")
        if "\\boxed{" not in response:
            errors.append(f"Row {i}, id {rid}: response has no \\boxed{{...}} final answer")

    got = set(got_ids)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    duplicate_count = len(got_ids) - len(got)
    if missing:
        errors.append(f"Missing ids: first 20 {missing[:20]} (total {len(missing)})")
    if extra:
        errors.append(f"Unexpected ids: first 20 {extra[:20]} (total {len(extra)})")
    if duplicate_count:
        errors.append(f"Duplicate id rows: {duplicate_count}")
    if len(rows) != len(problems):
        errors.append(f"CSV row count {len(rows)} != input problem count {len(problems)}")

    return len(errors) == 0, errors


# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------


SYSTEM_PROMPT_FREEFORM = r"""You are an expert mathematician solving a contest math problem.
Reason carefully, check your work, and finish without repeating yourself.

Final-answer rules are mandatory:
- Put the final answer on the LAST line only.
- Use exactly one final \boxed{...} expression on that last line.
- If the problem has multiple [ANS] blanks, put the answers in the same order, comma-separated inside one box.
- Use exact forms such as fractions, radicals, powers, intervals, ordered pairs, or symbolic expressions when exactness is natural.
- Do not round numerical answers unless the problem explicitly asks for rounded notation; if a decimal is necessary, give at least 12 significant digits.
- If an answer is an ordered pair, interval, or list item containing a comma, keep it inside parentheses/brackets so it remains one answer slot.
- Do not add labels such as "A.", "part a=", "x=", or units inside the final box unless the label/unit is required as the answer.
- Do not put any extra text, formula, or second box after the final \boxed{...} line."""

FREEFORM_AGGRESSIVE_NOTES = r"""

Additional answer-format discipline:
- If the problem asks for a general trigonometric solution such as sin(theta)=c, cos(theta)=c, or tan(theta)=c, prefer exact inverse-trig notation and periods inside the box, for example \boxed{\arctan(4.76), \pi}, unless the problem explicitly requires decimal radians.
- For numerical answers from statistics, regression, probability, normal/t/chi-square tests, temperatures, exponential models, or scientific data tables, carry enough precision through the calculation and put at least 10 significant digits in the final box.
- If the prompt says to round a repeating decimal, still include at least 6 decimal places unless it explicitly says "exactly 3 decimal places".
- For equations requested as answers, include the full equation exactly, including the right-hand side and multiplication signs when needed.
- For blanks that ask to choose from inline choices A., B., C., etc., put only the choice letter(s), not the choice text or a computed value.
- Before the final line, count the [ANS] blanks one last time. The final box must contain exactly that many top-level comma-separated answer items."""

FREEFORM_PRECISION_AUDIT_NOTES = r"""

Precision-audit mode:
- Treat every [ANS] blank as a separate graded field. Before the final box, make a checklist with one item per blank.
- For statistics, regression, confidence intervals, hypothesis tests, p-values, F/t/z/chi-square critical values, exponential models, and data-table arithmetic, do not stop at a rounded mental estimate. Recompute carefully and keep guard digits.
- If a blank asks for a test statistic, p-value, probability, regression coefficient, correlation, critical value, or interval endpoint, put a decimal with at least 8 significant digits unless the prompt explicitly demands fewer.
- If a blank asks for an option from choices shown in the prompt, output only the capital letter(s), not the numerical value or text of the option.
- For interval notation, preserve the requested open/closed endpoints and use -infinity/infinity if the starter wording asks for interval notation.
- For stem-and-leaf, design-generator, coding, or symbolic-list blanks, do not invent decimal approximations; output the exact symbols/leaves/letters requested.
- The final box must contain exactly the same number of top-level comma-separated items as there are [ANS] blanks."""

FREEFORM_COMPETITION_PRECISION_NOTES = r"""

Competition-precision mode:
- The contest judge compares against high-precision stored answers. Even if the word problem says "round to 4 decimals" or "use 3 decimal places", compute the unrounded value and put at least 8 significant digits in the final box when possible.
- This override matters most for probabilities, p-values, test statistics, confidence interval endpoints, correlations, regression coefficients, standard errors, and ANOVA quantities.
- If a slot asks for a letter choice, output the letter only; this precision override is only for numeric slots.
- Do not replace a numeric answer with an approximate table label. Carry guard digits and output the most precise decimal you can justify."""

FREEFORM_FORMULA_AUDIT_NOTES = r"""

Formula-audit mode:
- If a requested numerical value can be represented exactly as a formula from the given constants, prefer the exact formula over a rounded decimal.
- This is especially useful for exponential-growth/decay models, Newton-law temperature models, trigonometric degree-minute-second conversion, confidence-interval endpoints, standard errors, means, and proportions.
- Keep formulas simple and evaluator-friendly: use fractions, powers, \sqrt{...}, \sin(...), \cos(...), \ln(...), and ordinary arithmetic.
- Do not use undefined functions such as CDF, invT, qnorm, or software-specific notation in the final box.
- If the problem explicitly asks to round to a whole number or to choose an option letter, follow that instruction instead of leaving a formula.
- For multiple [ANS] blanks, count the blanks and put exactly that many formula/value/letter items in order inside the final box."""

FREEFORM_STATS_TABLE_NOTES = r"""

Statistics/precision table mode:
- Many blanks in these problems are graded numerically with very tight tolerance. Do not use rough table values or rounded mental estimates when a formula can be written exactly.
- For hypothesis tests, first identify z vs t vs F: known population sigma means z; unknown sigma with one sample means t with df=n-1; pooled two-sample t uses df=n1+n2-2; one-way ANOVA uses F with df_between=k-1 and df_within=N-k.
- Use interval notation exactly as requested, e.g. (-infinity, -2.2816) or (-infinity, -2.20119) U (2.20119, infinity).
- Useful critical values seen in this benchmark family:
  z_{0.10 upper}=1.2815515655446; z_{0.08 upper}=1.40507156030963; z_{0.05 upper}=1.64485362695147; z_{0.04 lower}=-1.75068607125217.
  t_{0.02 lower, df=13}=-2.2816; t_{0.0225 upper, df=14}=2.20119; t_{0.025 upper, df=18}=2.10092.
  F_{0.01 upper, df1=2, df2=15}=6.35886; F_{0.025 upper, df1=9, df2=9}=4.02599.
- For confidence intervals, output endpoints with at least 10 significant digits unless the prompt explicitly says fewer.
- For p-values, output at least 8 significant digits when possible.
- If the prompt asks for a choice among inline choices A., B., C., etc., output only the letter for that slot.
- If a blank asks for a trigonometric value and the prompt says use at least 5 decimals, an exact expression such as sin(43.5558333333333*pi/180) is safer than a rounded decimal."""

FREEFORM_GEOMETRY_AUDIT_NOTES = r"""

Geometry/vector audit mode:
- For travel, course-correction, bearing, force, or vector-resultant word problems, identify each displacement/vector length and the included angle before using the law of cosines or components.
- A turn of d degrees from the current/original course usually means the angle between the displacement vectors is d degrees; check this carefully before using 180-d.
- For ladder/right-triangle motion problems, compute the initial height and final height from the Pythagorean theorem, then subtract in the requested direction.
- Keep enough guard digits in square roots and trigonometric values before writing the final decimal."""

FREEFORM_COUNTING_AUDIT_NOTES = r"""

Counting-audit mode:
- For digit-counting, sticker-counting, page-numbering, or ID-number problems, first determine the inclusive first and last labels and the total number of labels.
- Count occurrences place by place when possible, and check the answer against a small manual block or complement count.
- For recurrence, sequence, or summation questions, verify whether the requested output is a single term, a count, or a sum over a range before writing the final integer.
- If an answer is a single integer, do not replace it with an explanation or formula in the final box."""

FREEFORM_CONCISE_COMPUTE_NOTES = r"""

Concise-compute mode:
- Do not restate or copy long data vectors, tables, or answer choices into the reasoning.
- Use compact arithmetic and sanity checks; avoid long narrative derivations.
- For statistics/data-table problems, identify the requested quantities, compute them directly, and keep guard digits.
- For multi-blank problems, write a short numbered scratch list with one result per blank, then the final boxed list.
- If the problem contains inline A./B./C. choices, output only the requested letter for those blanks.
- End as soon as the final boxed answer is complete; do not continue reasoning after the box."""

SYSTEM_PROMPT_MCQ = r"""You are an expert mathematician solving a multiple-choice math problem.
Reason carefully, compare all options, and finish without repeating yourself.

Final-answer rules are mandatory:
- Choose exactly one option letter.
- Put the final answer on the LAST line only.
- Use exactly one final boxed letter, e.g. \boxed{C}.
- Do not put any extra text, formula, or second box after the final \boxed{...} line."""


def option_labels(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def has_inline_choices(question: str) -> bool:
    """Detect free-form rows that embed A./B./C. choices in the prompt text."""
    return bool(
        re.search(r"\bA[\.)]\s", question)
        and re.search(r"\bB[\.)]\s", question)
        and re.search(r"\bC[\.)]\s", question)
    )


def expected_answer_count(item: Dict[str, Any]) -> Optional[int]:
    if item.get("options"):
        return 1
    slots = item.get("question", "").count("[ANS]")
    return slots if slots > 0 else None


def build_prompt_parts(item: Dict[str, Any]) -> Tuple[str, str]:
    q = str(item["question"])
    options = item.get("options") or []
    if options:
        labels = option_labels(len(options))
        options_text = "\n".join(f"{lab}. {str(opt).strip()}" for lab, opt in zip(labels, options))
        mcq_style = os.environ.get("CSE151B_MCQ_PROMPT_STYLE", "").lower()
        if mcq_style in {"valuefirst", "derive_first"}:
            user = (
                f"{q}\n\nOptions:\n{options_text}\n\n"
                "First solve the mathematical problem without choosing any letter. "
                "Write the derived target value, formula, or object in your reasoning. "
                "Then compare that derived result to the option texts exactly, including equivalent algebraic forms. "
                "If two options look close, substitute or simplify them against your derived result before choosing. "
                "Your final line must be exactly one boxed option letter, like \\boxed{C}."
            )
        elif mcq_style in {"value_elimination", "valueelim", "derive_eliminate"}:
            user = (
                f"{q}\n\nOptions:\n{options_text}\n\n"
                "First solve the mathematical problem without using the option letters. "
                "Derive the requested target value, formula, expression, set, or object as explicitly as possible. "
                "Then eliminate options one by one by comparing each option text to that derived target. "
                "Check exact equivalence: signs, constants, domains, branches, notation, simplification, expansion/factoring, and substitutions when useful. "
                "Do not choose an option because it looks close; choose only the option whose full text is mathematically identical to the derived target. "
                "Before the final line, state the surviving option and why the other close options fail. "
                "Your final line must be exactly one boxed option letter, like \\boxed{C}."
            )
        elif mcq_style in {"elimination", "audit"}:
            user = (
                f"{q}\n\nOptions:\n{options_text}\n\n"
                "Solve the problem by deriving the target quantity first, then compare it against every option. "
                "Be careful with equivalent algebraic forms, signs, domains, and notation traps. "
                "Before the final line, explicitly verify that the selected letter's option text matches your derived result. "
                "Your final line must be exactly one boxed option letter, like \\boxed{C}."
            )
        elif mcq_style in {"strict_derive_verify", "derive_verify"}:
            user = (
                f"{q}\n\nOptions:\n{options_text}\n\n"
                "Solve this multiple-choice math problem carefully.\n\n"
                "First, ignore the option letters and derive the target value, formula, expression, set, or object requested by the question.\n\n"
                "Then compare your derived result against every option one by one. For each option, check exact mathematical equivalence, including signs, constants, domains, branches, notation, simplification, factoring/expansion, and substitutions if useful.\n\n"
                "Do not choose a letter because it looks similar. Choose only the option whose full option text matches the derived result.\n\n"
                "If two options look close, simplify both or plug in a valid test value to distinguish them.\n\n"
                "Final line must contain exactly one boxed option letter, like \\boxed{C}."
            )
        else:
            user = (
                f"{q}\n\nOptions:\n{options_text}\n\n"
                "Solve the problem. Your final line must be exactly one boxed option letter, like \\boxed{C}."
            )
        return SYSTEM_PROMPT_MCQ, user

    n = q.count("[ANS]")
    inline_choice_note = ""
    if has_inline_choices(q):
        inline_choice_note = (
            "\n\nImportant: this problem includes answer choices written in the prompt "
            "(A., B., C., etc.). When a blank asks you to select, choose, match, "
            "or identify an option, output the option letter(s), not the option text "
            "or computed value. For numeric or symbolic blanks, output the value."
        )
    system = SYSTEM_PROMPT_FREEFORM
    prompt_style = os.environ.get("CSE151B_PROMPT_STYLE", "").lower()
    if prompt_style in {"aggressive_free", "aggressive"} or "aggressive" in prompt_style:
        system += FREEFORM_AGGRESSIVE_NOTES
    if prompt_style in {"precision_audit", "stats_precision", "audit_free"} or "precision" in prompt_style:
        system += FREEFORM_PRECISION_AUDIT_NOTES
    if prompt_style in {"competition_precision", "noround", "unrounded"} or "noround" in prompt_style or "unrounded" in prompt_style:
        system += FREEFORM_COMPETITION_PRECISION_NOTES
    if prompt_style in {"formula_audit", "exact_formula", "symbolic_numeric"} or "formula" in prompt_style:
        system += FREEFORM_FORMULA_AUDIT_NOTES
    if prompt_style in {"stats_table", "statistics_table"} or "stats" in prompt_style:
        system += FREEFORM_STATS_TABLE_NOTES
    if prompt_style in {"geometry_audit", "vector_audit"} or "geometry" in prompt_style or "vector" in prompt_style:
        system += FREEFORM_GEOMETRY_AUDIT_NOTES
    if prompt_style in {"counting_audit", "combinatorics_audit"} or "counting" in prompt_style:
        system += FREEFORM_COUNTING_AUDIT_NOTES
    if prompt_style in {"concise_compute", "direct_compute", "short_compute"} or "concise" in prompt_style or "direct" in prompt_style:
        system += FREEFORM_CONCISE_COMPUTE_NOTES

    if "concise" in prompt_style or "direct" in prompt_style:
        if n >= 2:
            user = (
                f"{q}{inline_choice_note}\n\nThis problem has {n} [ANS] blanks. "
                "Compute the requested answers directly without copying the full data/table. "
                f"Return exactly {n} answers in order inside one final box, comma-separated."
            )
        elif n == 1:
            user = (
                f"{q}{inline_choice_note}\n\nCompute the requested answer directly without copying the full data/table. "
                "Return exactly one final boxed answer."
            )
        else:
            user = (
                f"{q}{inline_choice_note}\n\nCompute the requested answer directly without copying the full data/table. "
                "Return exactly one final boxed answer."
            )
    elif n >= 2:
        user = (
            f"{q}{inline_choice_note}\n\nThis problem has {n} [ANS] blanks. "
            f"Solve all parts and put exactly {n} answers in order inside one final box, "
            "comma-separated. Before writing the final box, count the answers and make sure "
            f"there are exactly {n} top-level comma-separated items."
        )
    elif n == 1:
        user = f"{q}{inline_choice_note}\n\nSolve the problem and put the single final answer inside one final box."
    else:
        user = f"{q}{inline_choice_note}\n\nSolve the problem and put the final answer inside one final box."
    return system, user


def apply_chat_template(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -----------------------------------------------------------------------------
# Extraction, normalization, voting
# -----------------------------------------------------------------------------


def balanced_box_contents(text: str) -> List[Tuple[int, int, str]]:
    entries: List[Tuple[int, int, str]] = []
    needle = "\\boxed{"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        i = idx + len(needle)
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            entries.append((idx, i, text[idx + len(needle): i - 1].strip()))
        start = max(i, idx + 1)
    return entries


def extract_boxed_answer(text: str) -> str:
    """Match starter judger behavior: use content after last </think>, then last contiguous boxed group."""
    text = text or ""
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
    entries = balanced_box_contents(search_text)
    if not entries:
        entries = balanced_box_contents(text)
        search_text = text
    if not entries:
        return ""

    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = search_text[entries[j][1]:entries[j + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else:
            break
    return ", ".join(e[2] for e in last_group if e[2])


def split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def canonical_vote_key(ans: str, is_mcq: bool) -> str:
    ans = (ans or "").strip().strip("$ ")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\,", "")
    ans = re.sub(r"\s+", "", ans)
    if is_mcq:
        m = re.search(r"[A-Za-z]", ans)
        return m.group(0).upper() if m else ans.upper()
    parts = split_top_level_commas(ans)
    if len(parts) > 1:
        return ",".join(canonical_vote_key(x, False) for x in parts)
    return ans.lower()


def answer_count_ok(item: Dict[str, Any], ans: str) -> bool:
    exp = expected_answer_count(item)
    if exp is None:
        return True
    if item.get("options"):
        return len(canonical_vote_key(ans, True)) == 1
    return len(split_top_level_commas(ans)) == exp


def option_letter_ok(item: Dict[str, Any], ans: str) -> bool:
    if not item.get("options"):
        return True
    key = canonical_vote_key(ans, True)
    return key in set(option_labels(len(item.get("options") or [])))


def final_box(ans: str) -> str:
    ans = (ans or "").strip()
    if not ans:
        ans = "A" if re.fullmatch(r"[A-Za-z]?", ans) else "0"
    return f"\\boxed{{{ans}}}"


def sanitize_response(raw: str, chosen_answer: str) -> str:
    """Keep full trace but guarantee the last line is the chosen canonical final box."""
    raw = (raw or "").strip()
    canonical_last = final_box(chosen_answer)
    # Do not remove the raw output; the competition asks for full trace. Add a stable final line.
    if raw.endswith(canonical_last):
        return raw
    # Separate the appended box from any earlier boxed display math. The starter
    # judger merges contiguous boxes, which can duplicate multi-slot answers.
    return raw.rstrip() + "\n\nFinal answer:\n" + canonical_last


def forced_box_prefix(raw: str) -> str:
    """Prefix for a same-model continuation when thinking did not reach a boxed answer."""
    if "</think>" in (raw or ""):
        return "\n\n\\boxed{"
    return "\n</think>\n\n\\boxed{"


@dataclass
class CandidateGroup:
    vote_key: str
    answer: str
    responses: List[str]
    good_format: bool


def group_candidates(item: Dict[str, Any], responses: Sequence[str]) -> Dict[str, CandidateGroup]:
    is_mcq = bool(item.get("options"))
    groups: Dict[str, CandidateGroup] = {}
    for resp in responses:
        ans = extract_boxed_answer(resp)
        if not ans:
            continue
        key = canonical_vote_key(ans, is_mcq)
        good = answer_count_ok(item, ans) and option_letter_ok(item, ans)
        penalized_key = key if good else "__badformat__" + key
        if penalized_key not in groups:
            groups[penalized_key] = CandidateGroup(penalized_key, ans, [], good)
        groups[penalized_key].responses.append(resp)
    return groups


def choose_by_vote(item: Dict[str, Any], responses: Sequence[str]) -> Dict[str, Any]:
    groups = group_candidates(item, responses)
    is_mcq = bool(item.get("options"))
    if not groups:
        n = expected_answer_count(item) or 1
        fallback = "A" if is_mcq else ", ".join(["0"] * n)
        return {
            "answer": fallback,
            "response": final_box(fallback),
            "vote_key": canonical_vote_key(fallback, is_mcq),
            "votes": 0,
            "tied": False,
            "all_vote_counts": {},
        }

    def score(g: CandidateGroup) -> Tuple[int, int, int]:
        # More votes, valid format, shorter answer.
        return (len(g.responses), int(g.good_format), -len(g.answer))

    best_score = max(score(g) for g in groups.values())
    winners = [g for g in groups.values() if score(g) == best_score]
    # Choose a concise but non-empty trace among the winning group.
    winner = sorted(winners, key=lambda g: (len(g.answer), g.vote_key))[0]
    best_response = sorted(winner.responses, key=lambda r: (len(r) < 20, len(r)))[0]
    clean_key = winner.vote_key.replace("__badformat__", "")
    return {
        "answer": winner.answer,
        "response": sanitize_response(best_response, winner.answer),
        "vote_key": clean_key,
        "votes": len(winner.responses),
        "tied": len(winners) > 1,
        "all_vote_counts": {g.vote_key.replace("__badformat__", ""): len(g.responses) for g in groups.values()},
    }


# -----------------------------------------------------------------------------
# Difficulty and sample allocation
# -----------------------------------------------------------------------------


def is_hard(item: Dict[str, Any]) -> bool:
    q = item.get("question", "")
    qlow = q.lower()
    slots = q.count("[ANS]")
    if len(q) > 1100 or slots >= 3:
        return True
    if slots == 0 and not item.get("options"):
        return True
    hard_terms = [
        "prove", "olympiad", "analytic", "complex", "eigen", "matrix", "determinant",
        "integral", "differential", "probability", "geometry", "polynomial", "series",
        "limit", "recurrence", "modulo", "combin", "topolog", "group", "field",
    ]
    return any(t in qlow for t in hard_terms)


def sample_count(item: Dict[str, Any], args: argparse.Namespace) -> int:
    if args.samples is not None:
        return args.samples
    return args.samples_hard if is_hard(item) else args.samples_simple


# -----------------------------------------------------------------------------
# Model engines
# -----------------------------------------------------------------------------


class BaseEngine:
    tokenizer: Any

    def chat_prompt(self, system: str, user: str) -> str:
        return apply_chat_template(self.tokenizer, system, user)

    def generate_many(self, prompts: List[str], n: int, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> List[List[str]]:
        raise NotImplementedError


class VLLMEngine(BaseEngine):
    def __init__(self, args: argparse.Namespace):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.args = args
        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        quantization = None if args.quantization.lower() == "none" else args.quantization
        load_format = None if args.load_format.lower() == "none" else args.load_format
        llm_kwargs = dict(
            model=args.model,
            trust_remote_code=True,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_prefix_caching=args.enable_prefix_caching,
            seed=args.seed,
        )
        if quantization is not None:
            llm_kwargs["quantization"] = quantization
        if load_format is not None:
            llm_kwargs["load_format"] = load_format
        self.llm = LLM(**llm_kwargs)

    def generate_many(self, prompts: List[str], n: int, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> List[List[str]]:
        sp = self.SamplingParams(
            n=n,
            max_tokens=max_tokens or self.args.max_tokens,
            temperature=self.args.temperature if temperature is None else temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
            presence_penalty=self.args.presence_penalty,
        )
        outputs = self.llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        return [[o.text.strip() for o in out.outputs] for out in outputs]


class TransformersEngine(BaseEngine):
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.args = args
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )

    def generate_many(self, prompts: List[str], n: int, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> List[List[str]]:
        all_prompts: List[str] = []
        owners: List[int] = []
        for i, p in enumerate(prompts):
            for _ in range(n):
                all_prompts.append(p)
                owners.append(i)
        grouped: List[List[str]] = [[] for _ in prompts]
        bs = self.args.transformers_batch_size
        for start in range(0, len(all_prompts), bs):
            sub = all_prompts[start:start + bs]
            inputs = self.tokenizer(
                sub,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_model_len,
            ).to(self.model.device)
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.args.max_tokens,
                    temperature=self.args.temperature if temperature is None else temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    do_sample=(self.args.temperature if temperature is None else temperature) > 0,
                    repetition_penalty=self.args.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            for j, seq in enumerate(out):
                # For padded batches this is approximate but fine for generated suffix.
                cut = inputs["input_ids"].shape[1]
                text = self.tokenizer.decode(seq[cut:], skip_special_tokens=False).strip()
                grouped[owners[start + j]].append(text)
        return grouped


class PlaceholderEngine(BaseEngine):
    """No-model engine for CSV smoke tests only; not competitive."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tokenizer = None

    def chat_prompt(self, system: str, user: str) -> str:
        return user

    def generate_many(self, prompts: List[str], n: int, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> List[List[str]]:
        out = []
        for p in prompts:
            # Placeholder format only.
            ans = "A" if "Options:" in p else "0"
            out.append([f"Format-only placeholder.\n{final_box(ans)}" for _ in range(n)])
        return out


def make_engine(args: argparse.Namespace) -> BaseEngine:
    if args.placeholder_only:
        return PlaceholderEngine(args)
    if args.engine == "vllm":
        return VLLMEngine(args)
    if args.engine == "transformers":
        return TransformersEngine(args)
    try:
        return VLLMEngine(args)
    except Exception as e:
        print(f"[warn] vLLM init failed: {e}\n[warn] Falling back to Transformers.", file=sys.stderr)
        return TransformersEngine(args)


# -----------------------------------------------------------------------------
# Optional same-model selector for tied votes
# -----------------------------------------------------------------------------


def selector_prompt(item: Dict[str, Any], tied_answers: List[str]) -> Tuple[str, str]:
    q = str(item["question"])
    opts = item.get("options") or []
    if opts:
        labels = option_labels(len(opts))
        opt_text = "\n".join(f"{lab}. {opt}" for lab, opt in zip(labels, opts))
        q = f"{q}\n\nOptions:\n{opt_text}"
    cand = "\n".join(f"- {a}" for a in tied_answers)
    system = SYSTEM_PROMPT_MCQ if opts else SYSTEM_PROMPT_FREEFORM
    user = (
        f"{q}\n\nSeveral independent Qwen solutions disagreed. Candidate final answers:\n{cand}\n\n"
        "Solve independently, then choose the candidate that is most likely correct. "
        "Your final line must contain only one \\boxed{...} answer."
    )
    return system, user


def apply_selector_if_needed(item: Dict[str, Any], result: Dict[str, Any], responses: Sequence[str], engine: BaseEngine, args: argparse.Namespace) -> Dict[str, Any]:
    if not args.selector_on_tie or not result.get("tied"):
        return result
    groups = group_candidates(item, responses)
    if not groups:
        return result
    max_votes = max(len(g.responses) for g in groups.values())
    tied = [g for g in groups.values() if len(g.responses) == max_votes and g.good_format]
    if len(tied) <= 1:
        return result
    answers = [g.answer for g in tied]
    system, user = selector_prompt(item, answers)
    prompt = engine.chat_prompt(system, user)
    selector_out = engine.generate_many([prompt], n=1, max_tokens=min(args.max_tokens, args.selector_max_tokens), temperature=args.selector_temperature)[0][0]
    sel_ans = extract_boxed_answer(selector_out)
    sel_key = canonical_vote_key(sel_ans, bool(item.get("options")))
    for g in tied:
        if canonical_vote_key(g.answer, bool(item.get("options"))) == sel_key:
            # Keep original full solving trace for the selected answer group.
            best_response = sorted(g.responses, key=lambda r: (len(r) < 20, len(r)))[0]
            result.update({
                "answer": g.answer,
                "response": sanitize_response(best_response, g.answer),
                "vote_key": sel_key,
                "selector_used": True,
                "selector_response_preview": selector_out[:500],
            })
            return result
    # If selector picked a new answer, use selector trace itself; still Qwen-generated.
    if sel_ans:
        result.update({
            "answer": sel_ans,
            "response": sanitize_response(selector_out, sel_ans),
            "vote_key": sel_key,
            "selector_used": True,
            "selector_response_preview": selector_out[:500],
        })
    return result


# -----------------------------------------------------------------------------
# Public scoring
# -----------------------------------------------------------------------------


def extract_letter(text: str) -> str:
    ans = extract_boxed_answer(text)
    m = re.fullmatch(r"\s*([A-Za-z])\s*", ans or "")
    if m:
        return m.group(1).upper()
    m = re.search(r"[A-Za-z]", ans or "")
    if m:
        return m.group(0).upper()
    boxed = list(re.finditer(r"\\boxed\{\s*([A-Za-z])\s*\}", text))
    if boxed:
        return boxed[-1].group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def public_score(input_rows: Sequence[Dict[str, Any]], pred_by_id: Dict[int, str], repo_dir: str | Path) -> Optional[float]:
    if not input_rows or "answer" not in input_rows[0]:
        return None
    sys.path.insert(0, str(Path(repo_dir).resolve()))
    try:
        from judger import Judger  # type: ignore
    except Exception as e:
        print(f"[warn] Could not import judger.py from {repo_dir}: {e}", file=sys.stderr)
        return None
    judger = Judger(strict_extract=False)
    correct = 0
    total = 0
    for item in input_rows:
        rid = int(item["id"])
        if rid not in pred_by_id:
            continue
        resp = pred_by_id[rid]
        total += 1
        if item.get("options"):
            ok = extract_letter(resp) == str(item["answer"]).strip().upper()
        else:
            gold = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
            try:
                ok = bool(judger.auto_judge(pred=resp, gold=gold, options=[[]] * len(gold)))
            except Exception:
                ok = False
        correct += int(ok)
    return correct / total if total else None


# -----------------------------------------------------------------------------
# Main generation
# -----------------------------------------------------------------------------


def build_batches(remaining: List[Dict[str, Any]], args: argparse.Namespace) -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
    """Yield batches grouped by equal sample count, so vLLM n is constant per batch."""
    i = 0
    while i < len(remaining):
        n = sample_count(remaining[i], args)
        batch: List[Dict[str, Any]] = []
        while i < len(remaining) and len(batch) < args.batch_size:
            if sample_count(remaining[i], args) == n:
                batch.append(remaining[i])
                i += 1
            elif batch:
                break
            else:
                batch.append(remaining[i])
                i += 1
                break
        yield n, batch


def generate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    ids = [int(r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Input has duplicate ids; cannot create unambiguous submission.")

    done = load_checkpoint(args.checkpoint)
    print(f"Loaded {len(rows)} problems from {args.input}")
    print(f"Loaded {len(done)} checkpointed predictions from {args.checkpoint}")

    if args.write_from_checkpoint_only:
        missing = [rid for rid in ids if rid not in done]
        if missing:
            raise RuntimeError(f"Checkpoint missing {len(missing)} ids; first missing: {missing[:20]}")
        final_rows = [done[rid] for rid in ids]
        write_submission_csv(args.output, final_rows)
        ok, errs = validate_submission(args.input, args.output)
        print(f"Wrote {args.output}; validation={ok}")
        for e in errs[:20]:
            print("  -", e)
        return

    engine = make_engine(args)
    remaining = [r for r in rows if int(r["id"]) not in done]
    print(f"Remaining problems to generate: {len(remaining)}")

    for n, batch in build_batches(remaining, args):
        prompts: List[str] = []
        for item in batch:
            system, user = build_prompt_parts(item)
            prompts.append(engine.chat_prompt(system, user))
        t0 = time.time()
        grouped = engine.generate_many(prompts, n=n)
        elapsed = time.time() - t0
        print(f"batch={len(batch)} samples_each={n} elapsed={elapsed:.1f}s")

        if args.force_final_from_partial:
            forced_prompts: List[str] = []
            forced_targets: List[Tuple[int, int, str]] = []
            for item_idx, responses in enumerate(grouped):
                for response_idx, response in enumerate(responses):
                    if extract_boxed_answer(response):
                        continue
                    prefix = forced_box_prefix(response)
                    forced_prompts.append(prompts[item_idx] + response.rstrip() + prefix)
                    forced_targets.append((item_idx, response_idx, prefix))
            if forced_prompts:
                t1 = time.time()
                forced_outputs = engine.generate_many(
                    forced_prompts,
                    n=1,
                    max_tokens=args.force_final_max_tokens,
                    temperature=0.0,
                )
                forced_elapsed = time.time() - t1
                print(f"  forced_final={len(forced_prompts)} elapsed={forced_elapsed:.1f}s")
                for (item_idx, response_idx, prefix), outputs in zip(forced_targets, forced_outputs):
                    continuation = (outputs[0] if outputs else "").strip()
                    original = grouped[item_idx][response_idx].rstrip()
                    if continuation.startswith("\\boxed{"):
                        grouped[item_idx][response_idx] = original + "\n" + continuation
                    else:
                        grouped[item_idx][response_idx] = original + prefix + continuation

        for item, responses in zip(batch, grouped):
            result = choose_by_vote(item, responses)
            result = apply_selector_if_needed(item, result, responses, engine, args)
            rec = {
                "id": int(item["id"]),
                "response": result["response"],
                "extracted_answer": result["answer"],
                "vote_key": result["vote_key"],
                "votes": int(result.get("votes", 0)),
                "num_samples": len(responses),
                "tied": bool(result.get("tied", False)),
                "selector_used": bool(result.get("selector_used", False)),
                "all_vote_counts": result.get("all_vote_counts", {}),
                "is_mcq": bool(item.get("options")),
                "answer_slots": item.get("question", "").count("[ANS]"),
            }
            done[int(item["id"])] = rec
            append_jsonl(args.checkpoint, rec)

    final_rows = [done[int(r["id"])] for r in rows]
    write_submission_csv(args.output, final_rows)
    ok, errors = validate_submission(args.input, args.output)
    print(f"Wrote submission: {args.output} rows={len(final_rows)} validation={ok}")
    if errors:
        for e in errors[:30]:
            print("  -", e)
    if args.score_public:
        pred_by_id = {int(r["id"]): str(r["response"]) for r in final_rows}
        score = public_score(rows, pred_by_id, args.repo_dir)
        if score is not None:
            print(f"Public score: {score:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="private.jsonl or public.jsonl")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--checkpoint", default="submission_checkpoint.jsonl")
    p.add_argument("--repo-dir", default=".", help="directory containing judger.py")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--engine", choices=["auto", "vllm", "transformers"], default="auto")
    p.add_argument("--placeholder-only", action="store_true", help="make a valid-format placeholder CSV; NOT competitive")
    p.add_argument("--write-from-checkpoint-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=151)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples", type=int, default=None, help="override all sample counts")
    p.add_argument("--samples-simple", type=int, default=4)
    p.add_argument("--samples-hard", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)

    # vLLM
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization", default="none", help="e.g. none, bitsandbytes, fp8 depending on install/hardware")
    p.add_argument("--load-format", default="none", help="e.g. none, bitsandbytes")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=65536)
    p.add_argument("--enable-prefix-caching", action="store_true")

    # Transformers fallback
    p.add_argument("--transformers-batch-size", type=int, default=1)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--force-final-from-partial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-final-max-tokens", type=int, default=256)

    # same-model selector
    p.add_argument("--selector-on-tie", action="store_true")
    p.add_argument("--selector-max-tokens", type=int, default=8192)
    p.add_argument("--selector-temperature", type=float, default=0.2)

    p.add_argument("--score-public", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
