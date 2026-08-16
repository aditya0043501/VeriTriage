#!/usr/bin/env python3
"""
One-time benchmark: VeriTriage deterministic extraction vs. OpenAI LLM.

This script is NOT part of the VeriTriage product. It lives in benchmark/
and is never imported by main.py, extractors, or any production code.
The product remains 100% OpenAI-free.

Usage:
  cd benchmark
  python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  # Put your OpenAI API key in .env:
  echo "OPENAI_API_KEY=sk-..." > .env
  python llm_comparison.py

Output:
  - benchmark_results.md  (markdown summary table for pitch deck)
  - benchmark_results.json (full raw data)
"""

import os
import sys
import json
import time
import re
from pathlib import Path

# Load .env manually (no python-dotenv dependency needed)
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# Add backend to path for deterministic extraction imports
BACKEND_PATH = str(Path(__file__).parent.parent / "backend")
sys.path.insert(0, BACKEND_PATH)

from openai import OpenAI
from test_phrases import PHRASES, FIELD_DESCRIPTIONS

# Import our deterministic extractors
from extraction.rule_fallback import (
    extract_centor_fields,
    extract_wells_fields,
    extract_chadsvasc_fields,
)

# Module → extractor mapping
EXTRACTORS = {
    "sore_throat": extract_centor_fields,
    "leg_swelling": extract_wells_fields,
    "afib_stroke": extract_chadsvasc_fields,
}

# Context prefixes to simulate the conversation (same as audit)
CONTEXT_PREFIXES = {
    "sore_throat": "I have a sore throat",
    "leg_swelling": "my leg is swollen",
    "afib_stroke": "I have afib",
}

# Number of repeated runs for non-determinism check on hedged/unclear phrases
REPEAT_COUNT = 3

# Model to benchmark against
LLM_MODEL = "gpt-4o-mini"


def run_deterministic(field, phrase, module):
    """Run our deterministic extractor on a single phrase."""
    extract_fn = EXTRACTORS[module]
    prefix = CONTEXT_PREFIXES[module]
    combined = f"{prefix} {phrase}"
    try:
        extracted, unclear = extract_fn(combined, phrase, [field], last_asked_field=field)
        val = extracted.get(field)
        if val is True:
            return True
        elif val is False:
            return False
        else:
            return "unclear"
    except Exception as e:
        return f"error: {e}"


def build_llm_prompt(field, phrase):
    """Build a fair, non-leading prompt for the LLM."""
    desc = FIELD_DESCRIPTIONS[field]
    return (
        f"You are a clinical intake assistant. A patient has been asked about their medical history. "
        f"Based on the patient's response below, determine whether the following criterion is met.\n\n"
        f"Criterion: {desc}\n\n"
        f"Patient's response: \"{phrase}\"\n\n"
        f"Classify this as exactly one of:\n"
        f"- TRUE (the criterion is clearly met based on the patient's words)\n"
        f"- FALSE (the criterion is clearly NOT met based on the patient's words)\n"
        f"- UNCLEAR (the patient's response is ambiguous, hedged, or doesn't clearly indicate either way)\n\n"
        f"Respond with ONLY one word: TRUE, FALSE, or UNCLEAR. No explanation."
    )


def run_llm(client, field, phrase):
    """Send a single phrase to the LLM and parse the response."""
    prompt = build_llm_prompt(field, phrase)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # default temp — we want to measure real-world non-determinism
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip().upper()
        # Parse the response
        if "TRUE" in text and "FALSE" not in text and "UNCLEAR" not in text:
            return True
        elif "FALSE" in text and "UNCLEAR" not in text:
            return False
        elif "UNCLEAR" in text:
            return "unclear"
        elif text.startswith("TRUE"):
            return True
        elif text.startswith("FALSE"):
            return False
        else:
            return f"unparseable: {text[:50]}"
    except Exception as e:
        return f"error: {e}"


def run_llm_multiple(client, field, phrase, n=REPEAT_COUNT):
    """Run the LLM n times on the same phrase to check non-determinism."""
    results = []
    for _ in range(n):
        results.append(run_llm(client, field, phrase))
        time.sleep(0.3)  # rate limit courtesy
    return results


def is_correct(predicted, expected):
    """Check if a prediction matches the expected label."""
    if isinstance(predicted, str) and predicted.startswith(("error", "unparseable")):
        return False
    return predicted == expected


def main():
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found.")
        print("Create benchmark/.env with: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"VeriTriage vs. OpenAI {LLM_MODEL} Benchmark")
    print(f"Test set: {len(PHRASES)} phrases across {len(FIELD_DESCRIPTIONS)} fields")
    print(f"Non-determinism check: {REPEAT_COUNT}x on hedged/unclear phrases")
    print(f"{'=' * 70}")
    print()

    # Phase 1: Run deterministic extractor on all phrases
    print("Phase 1: Running deterministic extractor...")
    det_results = {}
    for i, (field, phrase, expected, category, module, is_hard) in enumerate(PHRASES):
        det_result = run_deterministic(field, phrase, module)
        det_results[(field, phrase)] = det_result
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(PHRASES)} done")
    print(f"  {len(PHRASES)}/{len(PHRASES)} done")

    # Phase 2: Run LLM on all phrases (single pass)
    print(f"\nPhase 2: Running LLM ({LLM_MODEL}) on all phrases...")
    llm_results = {}
    for i, (field, phrase, expected, category, module, is_hard) in enumerate(PHRASES):
        llm_result = run_llm(client, field, phrase)
        llm_results[(field, phrase)] = llm_result
        status = "OK" if not isinstance(llm_result, str) or not llm_result.startswith(("error", "unparseable")) else "ERR"
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(PHRASES)} done ({status})")
        time.sleep(0.2)  # rate limit courtesy
    print(f"  {len(PHRASES)}/{len(PHRASES)} done")

    # Phase 3: Non-determinism check — run hedged phrases 3x
    hedged_phrases = [(f, p, e, c, m, h) for f, p, e, c, m, h in PHRASES if c == "hedged"]
    print(f"\nPhase 3: Non-determinism check ({len(hedged_phrases)} hedged phrases × {REPEAT_COUNT} runs)...")
    llm_repeat_results = {}
    for i, (field, phrase, expected, category, module, is_hard) in enumerate(hedged_phrases):
        repeats = run_llm_multiple(client, field, phrase, REPEAT_COUNT)
        llm_repeat_results[(field, phrase)] = repeats
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(hedged_phrases)} done")
        time.sleep(0.2)
    print(f"  {len(hedged_phrases)}/{len(hedged_phrases)} done")

    # Phase 4: Score everything
    print(f"\nPhase 4: Scoring...")

    # Overall accuracy
    det_correct = 0
    llm_correct = 0
    total = len(PHRASES)

    # Hard subset (negation, hedged, family-history)
    hard_phrases = [(f, p, e, c, m, h) for f, p, e, c, m, h in PHRASES if h]
    det_hard_correct = 0
    llm_hard_correct = 0
    hard_total = len(hard_phrases)

    # Category breakdown
    cat_stats = {}
    for cat in ["yes", "no", "hedged"]:
        cat_stats[cat] = {"det_correct": 0, "llm_correct": 0, "total": 0}

    # Per-phrase detail for finding examples
    details = []

    for field, phrase, expected, category, module, is_hard in PHRASES:
        key = (field, phrase)
        det_val = det_results[key]
        llm_val = llm_results[key]

        det_ok = is_correct(det_val, expected)
        llm_ok = is_correct(llm_val, expected)

        if det_ok:
            det_correct += 1
        if llm_ok:
            llm_correct += 1

        if is_hard:
            if det_ok:
                det_hard_correct += 1
            if llm_ok:
                llm_hard_correct += 1

        cat_stats[category]["total"] += 1
        if det_ok:
            cat_stats[category]["det_correct"] += 1
        if llm_ok:
            cat_stats[category]["llm_correct"] += 1

        details.append({
            "field": field,
            "phrase": phrase,
            "expected": expected,
            "category": category,
            "module": module,
            "is_hard": is_hard,
            "det_result": det_val,
            "det_correct": det_ok,
            "llm_result": llm_val,
            "llm_correct": llm_ok,
        })

    # Non-determinism: how often did the LLM give different answers across runs?
    llm_nondeterminism_count = 0
    for key, repeats in llm_repeat_results.items():
        # Filter out errors
        valid = [r for r in repeats if not isinstance(r, str) or not r.startswith(("error", "unparseable"))]
        if len(valid) >= 2 and len(set(str(r) for r in valid)) > 1:
            llm_nondeterminism_count += 1

    # Deterministic non-determinism is always 0 (by definition)
    det_nondeterminism_count = 0

    # Find examples where LLM got it wrong and we got it right
    llm_wrong_det_right = [d for d in details if not d["llm_correct"] and d["det_correct"]]

    # Prefer hard cases for the pitch deck examples
    llm_wrong_det_right_hard = [d for d in llm_wrong_det_right if d["is_hard"]]
    llm_wrong_det_right_easy = [d for d in llm_wrong_det_right if not d["is_hard"]]

    # Pick top 5 examples (prioritize hard cases)
    examples = (llm_wrong_det_right_hard + llm_wrong_det_right_easy)[:5]

    # === Output markdown summary ===
    det_acc = det_correct / total * 100
    llm_acc = llm_correct / total * 100
    det_hard_acc = det_hard_correct / hard_total * 100 if hard_total > 0 else 0
    llm_hard_acc = llm_hard_correct / hard_total * 100 if hard_total > 0 else 0
    llm_nondet_rate = llm_nondeterminism_count / len(hedged_phrases) * 100 if hedged_phrases else 0

    md = []
    md.append("# VeriTriage vs. Generic LLM — Extraction Benchmark\n")
    md.append(f"**Test set:** {total} adversarial phrases across {len(FIELD_DESCRIPTIONS)} clinical criteria\n")
    md.append(f"**LLM:** OpenAI {LLM_MODEL} (temperature=0.7)\n")
    md.append(f"**Deterministic system:** VeriTriage rule-based extractor (no LLM)\n")
    md.append(f"**Non-determinism check:** {REPEAT_COUNT}× repeated runs on {len(hedged_phrases)} hedged phrases\n")
    md.append("")

    md.append("## Summary Table\n")
    md.append("| Metric | VeriTriage (deterministic) | Generic LLM (GPT-4o-mini) |")
    md.append("|--------|---------------------------|--------------------------|")
    md.append(f"| **Overall accuracy** | **{det_acc:.1f}%** ({det_correct}/{total}) | **{llm_acc:.1f}%** ({llm_correct}/{total}) |")
    md.append(f"| **Hard subset accuracy** (negation/hedged/family-history) | **{det_hard_acc:.1f}%** ({det_hard_correct}/{hard_total}) | **{llm_hard_acc:.1f}%** ({llm_hard_correct}/{hard_total}) |")
    md.append(f"| **Non-determinism rate** (hedged phrases) | **0.0%** (always identical) | **{llm_nondet_rate:.1f}%** ({llm_nondeterminism_count}/{len(hedged_phrases)}) |")
    md.append("")

    md.append("## Accuracy by Category\n")
    md.append("| Category | VeriTriage | Generic LLM |")
    md.append("|----------|-----------|------------|")
    for cat in ["yes", "no", "hedged"]:
        s = cat_stats[cat]
        det_pct = s["det_correct"] / s["total"] * 100 if s["total"] > 0 else 0
        llm_pct = s["llm_correct"] / s["total"] * 100 if s["total"] > 0 else 0
        label = {"yes": "Affirmative (yes)", "no": "Negative (no)", "hedged": "Hedged/uncertain"}[cat]
        md.append(f"| {label} | {det_pct:.1f}% ({s['det_correct']}/{s['total']}) | {llm_pct:.1f}% ({s['llm_correct']}/{s['total']}) |")
    md.append("")

    md.append("## Example Phrases: LLM Wrong, VeriTriage Right\n")
    if examples:
        md.append("These are cases where the generic LLM produced an incorrect classification")
        md.append("and VeriTriage's deterministic extractor produced the correct one.\n")
        for i, ex in enumerate(examples, 1):
            tag = " (hard case)" if ex["is_hard"] else ""
            md.append(f"### Example {i}{tag}")
            md.append(f"- **Field:** `{ex['field']}`")
            md.append(f"- **Patient said:** \"{ex['phrase']}\"")
            md.append(f"- **Correct answer:** `{ex['expected']}`")
            md.append(f"- **VeriTriage:** `{ex['det_result']}` ✅")
            md.append(f"- **LLM:** `{ex['llm_result']}` ❌")
            md.append("")
    else:
        md.append("(No cases found where LLM was wrong and VeriTriage was right.)\n")

    md.append("## Non-Determinism Detail\n")
    md.append(f"The LLM was run {REPEAT_COUNT}× on each of {len(hedged_phrases)} hedged/unclear phrases.")
    md.append(f"VeriTriage is deterministic — it always produces the same output for the same input.\n")
    if llm_nondeterminism_count > 0:
        md.append("Phrases where the LLM gave **different answers** across repeated runs:\n")
        for key, repeats in llm_repeat_results.items():
            valid = [r for r in repeats if not isinstance(r, str) or not r.startswith(("error", "unparseable"))]
            if len(valid) >= 2 and len(set(str(r) for r in valid)) > 1:
                field, phrase = key
                md.append(f"- \"{phrase}\" (`{field}`): {repeats}")
        md.append("")
    else:
        md.append("The LLM was consistent across all repeated runs (no non-determinism detected).\n")

    md.append("## Methodology\n")
    md.append("- Each phrase was sent to the LLM as a standalone classification task with a")
    md.append("  plain-language description of the clinical criterion (no leading questions,")
    md.append("  no scoring logic revealed).")
    md.append("- The deterministic extractor was run with the same phrase as the `current_input`")
    md.append("  and a context prefix matching the module (e.g. \"I have a sore throat\").")
    md.append("- Ground truth labels were established during the independent extraction audit.")
    md.append("- Hedged phrases were run 3× through the LLM at temperature=0.7 to measure")
    md.append("  non-determinism. VeriTriage is deterministic by construction (0% non-determinism).")
    md.append("- The LLM was instructed to respond with exactly one word: TRUE, FALSE, or UNCLEAR.")

    md_text = "\n".join(md)

    # Write outputs
    out_dir = Path(__file__).parent
    (out_dir / "benchmark_results.md").write_text(md_text)
    (out_dir / "benchmark_results.json").write_text(json.dumps({
        "model": LLM_MODEL,
        "total_phrases": total,
        "hard_subset_size": hard_total,
        "det_accuracy": det_acc,
        "llm_accuracy": llm_acc,
        "det_hard_accuracy": det_hard_acc,
        "llm_hard_accuracy": llm_hard_acc,
        "llm_nondeterminism_rate": llm_nondet_rate,
        "category_stats": cat_stats,
        "details": details,
        "llm_repeat_results": {f"{k[0]}|{k[1]}": [str(r) for r in v] for k, v in llm_repeat_results.items()},
    }, indent=2, default=str))

    # Print summary to console
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n{md_text}")
    print(f"\nResults written to:")
    print(f"  {out_dir / 'benchmark_results.md'}")
    print(f"  {out_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
