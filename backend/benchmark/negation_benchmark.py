#!/usr/bin/env python3
"""
Negation benchmark: VeriTriage deterministic extractor vs. GPT-4o.

This script is ISOLATED. It is never imported by main.py, extractors, or any
production code, and it is not wired into the main app or pipeline. It lives
entirely in backend/benchmark/ with its own requirements.txt.

Ground truth is the union of two existing test suites:
  1. The 153-phrase adversarial suite at benchmark/test_phrases.py (repo root).
  2. The negation regression suite at backend/tests/test_negation_regression.py.

For (2), this script does NOT hand-duplicate the phrases into a separate
hardcoded list. It reads the actual source of test_negation_regression.py at
run time and extracts each test method's phrase + expected outcome via a
small, deliberately narrow regex parser (see `_parse_negation_regression_file`
below). This means the two known call shapes used in that file
  - extract_{centor,wells,chadsvasc}_fields("", "<phrase>", ["<field>"], ...)
  - _is_negated_context("<phrase>", "<keyword>") / _positive_keyword_hit("<phrase>", ["<keyword>"])
are supported. If a new test method uses a different shape, the parser will
print a warning and skip it rather than silently mis-extracting it or
crashing the whole run. A small metadata table (module / category bucket /
plain-language description for the GPT prompt) is still maintained by hand
per test function name, since a docstring/assert alone can't tell us which
clinical module a raw keyword check belongs to — this is documented inline.

For our own extractor, this script calls the ACTUAL production functions in
extraction/rule_fallback.py (extract_centor_fields, extract_wells_fields,
extract_chadsvasc_fields, _is_negated_context, _positive_keyword_hit) — not a
reimplementation.

Usage:
  cd backend/benchmark
  python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-...
  python negation_benchmark.py

Output (written to backend/benchmark/):
  - negation_benchmark_results.md   (markdown summary table)
  - negation_benchmark_results.csv  (raw per-phrase results)
"""

import os
import sys
import csv
import time
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

THIS_DIR = Path(__file__).resolve().parent          # backend/benchmark
BACKEND_ROOT = THIS_DIR.parent                       # backend
REPO_ROOT = BACKEND_ROOT.parent                      # VeriTriage
TOP_LEVEL_BENCHMARK = REPO_ROOT / "benchmark"         # benchmark/ (153-phrase suite)
NEGATION_TEST_FILE = BACKEND_ROOT / "tests" / "test_negation_regression.py"

sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(TOP_LEVEL_BENCHMARK))

from extraction.rule_fallback import (  # noqa: E402
    extract_centor_fields,
    extract_wells_fields,
    extract_chadsvasc_fields,
    _is_negated_context,
    _positive_keyword_hit,
)
from test_phrases import PHRASES, FIELD_DESCRIPTIONS  # noqa: E402  (153-phrase suite)

EXTRACT_FNS = {
    "sore_throat": extract_centor_fields,
    "leg_swelling": extract_wells_fields,
    "afib_stroke": extract_chadsvasc_fields,
}

CONTEXT_PREFIXES = {
    "sore_throat": "I have a sore throat",
    "leg_swelling": "my leg is swollen",
    "afib_stroke": "I have afib",
}

GPT_MODEL = "gpt-4o"
GPT_RUNS_PER_PHRASE = 3
GPT_TEMPERATURE = 0.7  # non-zero so repeated runs can reveal real consistency behavior

# Current OpenAI pricing (USD per 1M tokens) for gpt-4o, standard tier.
# Source: https://developers.openai.com/api/docs/pricing (checked at script-write time).
# Update these constants if pricing changes before re-running.
PRICE_PER_1M_INPUT = 2.50
PRICE_PER_1M_OUTPUT = 10.00

# The five report-facing category buckets requested for this benchmark.
BUCKETS = [
    "pseudo-negation",
    "ruled-out family",
    "scope-termination",
    "plain negation",
    "uncertainty/hedged",
]

# ---------------------------------------------------------------------------
# Hand-maintained metadata for the negation-regression test cases, keyed by
# test function name. The phrase text and expected True/False/unclear value
# are NOT hardcoded here — those are parsed live from the test file's source
# below. This table only supplies what can't be inferred from an assert
# statement: which clinical module a raw keyword check belongs to, which of
# the 5 report buckets it falls into, and a plain-language description for
# the GPT prompt (for cases that don't map to an existing schema field with
# an entry already in FIELD_DESCRIPTIONS).
# ---------------------------------------------------------------------------
NEGATION_CASE_METADATA = {
    "test_no_evidence_of_extension_does_not_negate_calf": {
        "module": "leg_swelling",
        "bucket": "pseudo-negation",
        "description": "The patient's calf is currently swollen or affected — not merely mentioned in passing as part of an unrelated finding.",
    },
    "test_no_significant_change_does_not_negate_swelling": {
        "module": "sore_throat",
        "bucket": "pseudo-negation",
        "description": "The patient currently has swelling in their tonsils/throat area.",
    },
    "test_ruled_out_stroke_is_false": {
        "module": "afib_stroke",
        "bucket": "ruled-out family",
        "description": None,  # uses existing FIELD_DESCRIPTIONS["stroke_tia_history"]
    },
    "test_not_been_ruled_out_is_unclear": {
        "module": "afib_stroke",
        "bucket": "ruled-out family",
        "description": None,  # uses existing FIELD_DESCRIPTIONS["stroke_tia_history"]
    },
    "test_but_preserves_fever_after_negated_cough": {
        "module": "sore_throat",
        "bucket": "scope-termination",
        "description": None,  # uses existing FIELD_DESCRIPTIONS["fever"]
    },
    "test_not_present_directly_negates_blood_clot": {
        "module": "leg_swelling",
        "bucket": "plain negation",
        "description": "The patient currently has symptoms suggestive of a blood clot in the leg.",
    },
    "test_secondary_to_does_not_pull_earlier_concern_into_negation": {
        "module": "leg_swelling",
        "bucket": "scope-termination",
        "description": "The patient currently has symptoms suggestive of a blood clot in the leg.",
    },
}

_MODULE_BY_EXTRACT_FN = {
    "centor": "sore_throat",
    "wells": "leg_swelling",
    "chadsvasc": "afib_stroke",
}


def _parse_negation_regression_file(path: Path):
    """Parse test_negation_regression.py source to extract (phrase, field,
    expected, kind) for each test method. See module docstring for scope
    and limitations of this parser."""
    if not path.exists():
        print(f"WARNING: negation regression test file not found at {path}; "
              f"skipping that half of the ground-truth set.", file=sys.stderr)
        return []

    src = path.read_text()
    method_pattern = re.compile(
        r"def (test_\w+)\(self\):\n(.*?)(?=\n    def |\nif __name__)",
        re.DOTALL,
    )

    cases = []
    for m in method_pattern.finditer(src):
        name = m.group(1)
        body = m.group(2)
        parsed = _parse_single_negation_case(name, body)
        if parsed is None:
            print(f"WARNING: could not parse negation regression test "
                  f"'{name}' with the known call shapes; skipping it.", file=sys.stderr)
            continue
        meta = NEGATION_CASE_METADATA.get(name)
        if meta is None:
            print(f"WARNING: no metadata entry for negation regression test "
                  f"'{name}'; skipping it.", file=sys.stderr)
            continue
        parsed.update(meta)
        parsed["source"] = "negation-regression"
        parsed["test_name"] = name
        cases.append(parsed)
    return cases


def _parse_single_negation_case(name: str, body: str):
    # Shape 1: extract_{centor,wells,chadsvasc}_fields("", "<phrase>", ["<field>"], ...)
    m = re.search(
        r'extract_(centor|wells|chadsvasc)_fields\(\s*"([^"]*)",\s*"([^"]*)",\s*\[\s*"([a-zA-Z_]+)"\s*\]',
        body,
    )
    if m:
        fn_key, _combined_lit, phrase, field = m.groups()
        module = _MODULE_BY_EXTRACT_FN[fn_key]
        if re.search(rf'result\.get\("{re.escape(field)}"\)\s+is\s+True', body):
            expected = True
        elif re.search(rf'result\.get\("{re.escape(field)}"\)\s+is\s+False', body):
            expected = False
        elif re.search(rf'result\.get\("{re.escape(field)}"\)\s+is\s+None', body):
            expected = "unclear"
        else:
            return None
        return {
            "phrase": phrase,
            "field": field,
            "expected": expected,
            "kind": "field_extraction",
        }

    # Shape 2: _is_negated_context("<phrase>", "<keyword>") — negated => concept absent.
    m = re.search(r'_is_negated_context\(\s*"([^"]*)",\s*"([^"]*)"\s*\)', body)
    if m:
        phrase, keyword = m.groups()
        tail = body[m.end():m.end() + 20]
        tm = re.match(r'\s*is\s*(True|False)', tail)
        if not tm:
            return None
        negated = tm.group(1) == "True"
        return {
            "phrase": phrase,
            "field": keyword,
            "expected": (not negated),
            "kind": "is_negated_context",
        }

    # Shape 3: _positive_keyword_hit("<phrase>", ["<keyword>"]) — hit => concept present.
    m = re.search(r'_positive_keyword_hit\(\s*"([^"]*)",\s*\[\s*"([^"]*)"\s*\]\s*\)', body)
    if m:
        phrase, keyword = m.groups()
        tail = body[m.end():m.end() + 20]
        tm = re.match(r'\s*is\s*(True|False)', tail)
        if not tm:
            return None
        present = tm.group(1) == "True"
        return {
            "phrase": phrase,
            "field": keyword,
            "expected": present,
            "kind": "positive_keyword_hit",
        }

    return None


def _bucket_for_153_case(category: str) -> str:
    # The 153-phrase suite's own categories ("yes" / "no" / "hedged") don't
    # distinguish pseudo-negation / ruled-out / scope-termination from plain
    # polarity — those complex negation forms live only in the negation
    # regression suite. "yes" and "no" both fold into "plain negation" here
    # (i.e. "plain polarity, not adversarial negation"); "hedged" maps to
    # "uncertainty/hedged".
    if category == "hedged":
        return "uncertainty/hedged"
    return "plain negation"


def build_ground_truth():
    """Combine the 153-phrase suite and the negation regression suite into a
    single deduplicated ground-truth list. Returns (cases, dedup_report)."""
    cases = []

    for field, phrase, expected, category, module, is_hard in PHRASES:
        cases.append({
            "source": "153-suite",
            "field": field,
            "phrase": phrase,
            "expected": expected,
            "bucket": _bucket_for_153_case(category),
            "module": module,
            "kind": "field_extraction",
            "description": None,  # uses FIELD_DESCRIPTIONS[field]
            "test_name": None,
        })

    cases.extend(_parse_negation_regression_file(NEGATION_TEST_FILE))

    seen = set()
    deduped = []
    n_dupes = 0
    for c in cases:
        key = (c["module"], c["field"], c["phrase"].strip().lower())
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        deduped.append(c)

    return deduped, {
        "total_before_dedup": len(cases),
        "n_153_suite": sum(1 for c in cases if c["source"] == "153-suite"),
        "n_negation_regression": sum(1 for c in cases if c["source"] == "negation-regression"),
        "n_duplicates_removed": n_dupes,
        "total_after_dedup": len(deduped),
    }


def get_description(case) -> str:
    if case["description"]:
        return case["description"]
    return FIELD_DESCRIPTIONS.get(case["field"], f"The patient has: {case['field']}.")


# ---------------------------------------------------------------------------
# Our deterministic extractor (actual production code path)
# ---------------------------------------------------------------------------

def run_our_extractor(case) -> object:
    """Run the phrase through the ACTUAL production rule_fallback functions."""
    module = case["module"]
    phrase = case["phrase"]
    field = case["field"]
    kind = case["kind"]

    try:
        if kind == "field_extraction":
            extract_fn = EXTRACT_FNS[module]
            combined = f"{CONTEXT_PREFIXES[module]} {phrase}"
            extracted, _unclear = extract_fn(combined, phrase, [field], last_asked_field=field)
            val = extracted.get(field)
            if val is True:
                return True
            elif val is False:
                return False
            else:
                return "unclear"
        elif kind == "is_negated_context":
            negated = _is_negated_context(phrase, field)
            return not negated
        elif kind == "positive_keyword_hit":
            return _positive_keyword_hit(phrase, [field])
        else:
            return f"error: unknown kind {kind}"
    except Exception as e:
        return f"error: {e}"


# ---------------------------------------------------------------------------
# GPT-4o extraction (plain, non-leading prompt; structured JSON output)
# ---------------------------------------------------------------------------

def build_gpt_prompt(case) -> str:
    desc = get_description(case)
    return (
        f"A patient was asked about their medical history or symptoms. "
        f"Based ONLY on the patient's response below, determine whether the "
        f"following statement is true.\n\n"
        f"Statement: {desc}\n\n"
        f"Patient's response: \"{case['phrase']}\"\n\n"
        f"Respond with a JSON object of the form "
        f'{{"answer": "TRUE"}} or {{"answer": "FALSE"}} or {{"answer": "UNCLEAR"}}.\n'
        f"Use UNCLEAR only if the response is genuinely ambiguous, hedged, or "
        f"doesn't indicate either way. Respond with JSON only, no other text."
    )


def run_gpt_once(client, case):
    """Single GPT-4o call. Returns (parsed_value, prompt_tokens, completion_tokens)."""
    prompt = build_gpt_prompt(case)
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=GPT_TEMPERATURE,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        try:
            parsed = json.loads(text)
            answer = str(parsed.get("answer", "")).strip().upper()
        except (json.JSONDecodeError, AttributeError):
            answer = text.strip().upper() if text else ""

        if answer == "TRUE":
            value = True
        elif answer == "FALSE":
            value = False
        elif answer == "UNCLEAR":
            value = "unclear"
        else:
            value = f"unparseable: {str(text)[:60]}"
        return value, prompt_tokens, completion_tokens
    except Exception as e:
        return f"error: {e}", 0, 0


def is_correct(predicted, expected) -> bool:
    if isinstance(predicted, str) and predicted.startswith(("error", "unparseable")):
        return False
    return predicted == expected


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "This benchmark requires a real OpenAI API key to run the GPT-4o "
            "comparison. Set it and re-run, e.g.:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  python3 negation_benchmark.py\n"
            "Refusing to run rather than silently skipping the GPT half of "
            "the comparison.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: the 'openai' package is not installed.\n"
            "Install this benchmark's isolated requirements first:\n"
            "  pip install -r backend/benchmark/requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("Building combined ground-truth set...")
    cases, dedup_report = build_ground_truth()
    print(f"  153-phrase suite:        {dedup_report['n_153_suite']}")
    print(f"  Negation regression:     {dedup_report['n_negation_regression']}")
    print(f"  Total before dedup:      {dedup_report['total_before_dedup']}")
    print(f"  Duplicates removed:      {dedup_report['n_duplicates_removed']}")
    print(f"  Final combined count:    {dedup_report['total_after_dedup']}")
    print()

    total = len(cases)

    # --- Phase 1: our deterministic extractor ---
    print(f"Phase 1: running our deterministic extractor on {total} phrases...")
    t0 = time.time()
    for c in cases:
        c["our_output"] = run_our_extractor(c)
        c["our_correct"] = is_correct(c["our_output"], c["expected"])
    our_wall_time = time.time() - t0
    print(f"  done in {our_wall_time:.3f}s")
    print()

    # --- Phase 2: GPT-4o, 3 runs per phrase ---
    print(f"Phase 2: running GPT-4o ({GPT_RUNS_PER_PHRASE}x per phrase, "
          f"{total * GPT_RUNS_PER_PHRASE} total calls)...")
    total_prompt_tokens = 0
    total_completion_tokens = 0
    t0 = time.time()
    for i, c in enumerate(cases):
        runs = []
        for _ in range(GPT_RUNS_PER_PHRASE):
            value, p_tok, c_tok = run_gpt_once(client, c)
            runs.append(value)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            time.sleep(0.15)
        c["gpt_runs"] = runs
        valid_runs = [r for r in runs if not (isinstance(r, str) and r.startswith(("error", "unparseable")))]
        c["gpt_consistent"] = len(valid_runs) == len(runs) and len(set(str(r) for r in valid_runs)) <= 1
        # For accuracy scoring, use majority vote across the 3 runs; ties fall to the first valid run.
        if valid_runs:
            counts = defaultdict(int)
            for r in valid_runs:
                counts[str(r)] += 1
            majority_str = max(counts.items(), key=lambda kv: kv[1])[0]
            majority_val = {"True": True, "False": False, "unclear": "unclear"}.get(majority_str, valid_runs[0])
        else:
            majority_val = runs[0] if runs else "error: no runs"
        c["gpt_output"] = majority_val
        c["gpt_correct"] = is_correct(majority_val, c["expected"])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{total} phrases done")
    gpt_wall_time = time.time() - t0
    print(f"  done in {gpt_wall_time:.1f}s ({total} phrases x {GPT_RUNS_PER_PHRASE} runs)")
    print()

    # --- Phase 3: scoring ---
    print("Phase 3: scoring...")

    our_correct_n = sum(1 for c in cases if c["our_correct"])
    gpt_correct_n = sum(1 for c in cases if c["gpt_correct"])
    our_acc = our_correct_n / total * 100 if total else 0.0
    gpt_acc = gpt_correct_n / total * 100 if total else 0.0

    gpt_consistent_n = sum(1 for c in cases if c["gpt_consistent"])
    gpt_consistency_rate = gpt_consistent_n / total * 100 if total else 0.0

    bucket_stats = {b: {"total": 0, "our_correct": 0, "gpt_correct": 0} for b in BUCKETS}
    for c in cases:
        b = c["bucket"]
        if b not in bucket_stats:
            bucket_stats[b] = {"total": 0, "our_correct": 0, "gpt_correct": 0}
        bucket_stats[b]["total"] += 1
        if c["our_correct"]:
            bucket_stats[b]["our_correct"] += 1
        if c["gpt_correct"]:
            bucket_stats[b]["gpt_correct"] += 1

    cost_usd = (
        total_prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT
        + total_completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT
    )

    # --- Output: markdown ---
    md = []
    md.append("# VeriTriage Deterministic Extractor vs. GPT-4o — Negation Benchmark\n")
    md.append(f"**Ground truth:** {total} phrases "
              f"({dedup_report['n_153_suite']} from the 153-phrase suite + "
              f"{dedup_report['n_negation_regression']} from the negation regression suite, "
              f"{dedup_report['n_duplicates_removed']} duplicates removed)\n")
    md.append(f"**GPT model:** {GPT_MODEL} (temperature={GPT_TEMPERATURE}, "
              f"{GPT_RUNS_PER_PHRASE} runs per phrase, majority vote used for accuracy)\n")
    md.append(f"**Our system:** VeriTriage deterministic rule-based extractor "
              f"(`extraction/rule_fallback.py`, production code path, no LLM)\n")
    md.append("")

    md.append("## Summary\n")
    md.append("| Metric | VeriTriage (deterministic) | GPT-4o |")
    md.append("|---|---|---|")
    md.append(f"| Overall accuracy | {our_acc:.1f}% ({our_correct_n}/{total}) | {gpt_acc:.1f}% ({gpt_correct_n}/{total}) |")
    md.append(f"| Consistency across {GPT_RUNS_PER_PHRASE} repeated runs (identical input) | 100.0% (deterministic by construction) | {gpt_consistency_rate:.1f}% ({gpt_consistent_n}/{total}) |")
    md.append(f"| Wall-clock time | {our_wall_time:.3f}s | {gpt_wall_time:.1f}s |")
    md.append(f"| Estimated cost (this run) | $0.00 | ${cost_usd:.4f} |")
    md.append("")

    md.append("## Accuracy by Category\n")
    md.append("| Category | VeriTriage | GPT-4o | N |")
    md.append("|---|---|---|---|")
    for b in BUCKETS:
        s = bucket_stats.get(b, {"total": 0, "our_correct": 0, "gpt_correct": 0})
        n = s["total"]
        our_pct = s["our_correct"] / n * 100 if n else float("nan")
        gpt_pct = s["gpt_correct"] / n * 100 if n else float("nan")
        our_str = f"{our_pct:.1f}% ({s['our_correct']}/{n})" if n else "n/a (0 cases)"
        gpt_str = f"{gpt_pct:.1f}% ({s['gpt_correct']}/{n})" if n else "n/a (0 cases)"
        md.append(f"| {b} | {our_str} | {gpt_str} | {n} |")
    md.append("")

    md.append("## Cost Detail\n")
    md.append(f"- Total GPT-4o API calls: {total * GPT_RUNS_PER_PHRASE}")
    md.append(f"- Total prompt tokens: {total_prompt_tokens:,}")
    md.append(f"- Total completion tokens: {total_completion_tokens:,}")
    md.append(f"- Pricing used: ${PRICE_PER_1M_INPUT:.2f}/1M input tokens, "
              f"${PRICE_PER_1M_OUTPUT:.2f}/1M output tokens (gpt-4o standard tier)")
    md.append(f"- **Total estimated cost: ${cost_usd:.4f}**")
    md.append("")

    md.append("## Methodology Notes\n")
    md.append("- Ground truth combines two existing, independently-authored test suites; "
              "see the script docstring for exactly how the negation-regression suite's "
              "phrases were parsed from its test file.")
    md.append("- Our extractor was called directly (`extract_centor_fields`, "
               "`extract_wells_fields`, `extract_chadsvasc_fields`, `_is_negated_context`, "
               "`_positive_keyword_hit`) — this is the same code path used in production, "
               "not a reimplementation.")
    md.append("- GPT-4o was given a plain, non-leading description of the same clinical "
               "concept our extractor checks, with no hint about negation, hedging, or any "
               "other linguistic phenomenon being tested.")
    md.append("- GPT-4o was run 3 times per phrase at temperature=0.7 to measure real "
               "run-to-run consistency, not just single-sample accuracy. Accuracy scoring "
               "uses the majority vote across the 3 runs; consistency scoring checks whether "
               "all 3 runs agreed with each other, independent of correctness.")
    md.append("- Numbers are reported as measured, without adjustment.")

    md_text = "\n".join(md)

    md_path = THIS_DIR / "negation_benchmark_results.md"
    md_path.write_text(md_text)

    # --- Output: raw CSV ---
    csv_path = THIS_DIR / "negation_benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phrase", "category", "expected", "our_output",
            "gpt_run1", "gpt_run2", "gpt_run3", "gpt_consistent",
            "our_correct", "gpt_correct",
        ])
        for c in cases:
            runs = c["gpt_runs"] + [""] * (GPT_RUNS_PER_PHRASE - len(c["gpt_runs"]))
            writer.writerow([
                c["phrase"],
                c["bucket"],
                c["expected"],
                c["our_output"],
                runs[0], runs[1], runs[2],
                c["gpt_consistent"],
                c["our_correct"],
                c["gpt_correct"],
            ])

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 70}\n")
    print(md_text)
    print(f"\nResults written to:\n  {md_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
