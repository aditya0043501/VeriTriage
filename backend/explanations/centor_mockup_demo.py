#!/usr/bin/env python3
"""
Stage 5 worked mockup: Modified (McIsaac) Centor score explanation walkthrough.

Same deterministic, template-only approach as Stage 4 (wells_mockup_demo.py).
NOT wired into main.py or the production pipeline. Runs the ACTUAL scoring
engine (calculate_centor_score) on real sample data, then assembles the
patient-facing explanation from:
  - explanations/templates.json  (human-reviewed sentences; empty slots fall
    back to clearly-flagged [PLACEHOLDER] text for this mockup ONLY)
  - probability_mapping.py       (published, cited probability ranges only)
  - diagram_generator.py         (deterministic SVG, from rule-engine output)
  - disclaimer.py                (fixed disclaimer text)

Citation labeling (same pattern as Wells, see docs/clinical-explanation-
sources.md §2.3 for the full discrepancy-check writeup):
  - "Probability estimate source" -> McIsaac 2004 JAMA (the per-score-level
    probability table; used by probability_mapping.get_centor_probability())
  - "Scoring criteria source" -> the citation already returned by
    calculate_centor_score(), i.e. Centor 1981 (original derivation) +
    McIsaac 1998 CMAJ (age-modifier derivation) — a DIFFERENT pair of
    papers from the 2004 JAMA validation study used for the probability
    number. This is intentional (same pattern as Wells' 2003 NEJM/2006 JAMA
    split) but is called out explicitly so the two citations don't read as
    contradictory or like the same source being paraphrased two ways.

Two scenarios, matching the difficulty spread requested for Stage 5:
  1. Fever — an easy, self-evident criterion patients describe naturally.
  2. Tender anterior cervical adenopathy — a harder-to-self-report
     criterion; patients don't use this term, so the "you told us" text
     reflects a scaffolded question ("swollen or tender under your jaw"),
     the same pattern used for Wells' "collateral veins" mockup. This
     scenario also carries a non-zero age modifier (-1, age 45+), which
     is NOT a simple present/absent criterion, to exercise the new
     `match_text_override` support in diagram_generator.py.

Output: two markdown files + one SVG per matched criterion/modifier,
written to backend/explanations/mockup_output/.
"""

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = THIS_DIR.parent
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scoring.centor_score import calculate_centor_score  # noqa: E402
from explanations.probability_mapping import get_centor_probability, UncitedScoreError  # noqa: E402
from explanations.diagram_generator import generate_criterion_diagram  # noqa: E402
from explanations.disclaimer import EXPLANATION_DISCLAIMER  # noqa: E402

TEMPLATES_PATH = REPO_ROOT / "explanations" / "templates.json"
OUT_DIR = THIS_DIR / "mockup_output"

# Maps calculate_centor_score() boolean parameter names to (breakdown label,
# templates.json key). Centor's scoring function does not expose a CRITERIA
# list like wells_score.py, so this mapping is maintained here only, for the
# mockup layer — it does not change scoring logic.
CENTOR_BOOL_CRITERIA = [
    ("fever", "Fever", "fever"),
    ("absence_of_cough", "Absence of cough", "absence_of_cough"),
    ("tender_cervical_nodes", "Tender neck lymph nodes", "tender_cervical_nodes"),
    ("tonsillar_exudate", "Tonsillar exudate/swelling", "tonsillar_exudate"),
]

# ---------------------------------------------------------------------------
# Scenario 1: fever (easy — patients describe this in their own words).
# "I've had a fever, it's been running around 101°F."
# fever=True, everything else False, age=30 (no age adjustment).
# Score = 1 (low tier).
# ---------------------------------------------------------------------------
SCENARIO_FEVER = {
    "name": "Fever (easy, self-evident criterion)",
    "output_filename": "explanations_mockup_centor_fever.md",
    "patient_words": {
        "fever": "I've had a fever, it's been running around 101\u00b0F",
    },
    "criteria": {
        "fever": True,
        "absence_of_cough": False,
        "tender_cervical_nodes": False,
        "tonsillar_exudate": False,
    },
    "age": 30,
}

# ---------------------------------------------------------------------------
# Scenario 2: tender anterior cervical adenopathy (harder to self-report).
# Patients don't naturally say "anterior cervical adenopathy" — this
# requires a scaffolded question ("Do you feel any swollen or tender lumps
# on the front of your neck, under your jaw?"), same pattern as Wells'
# "collateral veins" scenario.
# "Yeah, when I press on the front of my neck under my jaw, it feels
#  swollen and sore."
# tender_cervical_nodes=True, everything else False, age=50 (-1 point).
# Score = 1 + (-1) = 0 (low tier, the "-1 or 0" bracket).
# ---------------------------------------------------------------------------
SCENARIO_TENDER_NODES = {
    "name": "Tender anterior cervical adenopathy (harder-to-self-report criterion)",
    "output_filename": "explanations_mockup_centor_tender_nodes.md",
    "patient_words": {
        "tender_cervical_nodes": "yeah, when I press on the front of my neck under my jaw, it feels swollen and sore",
    },
    "criteria": {
        "fever": False,
        "absence_of_cough": False,
        "tender_cervical_nodes": True,
        "tonsillar_exudate": False,
    },
    "age": 50,
}

SCENARIOS = [SCENARIO_FEVER, SCENARIO_TENDER_NODES]


def load_template(module_key: str, criterion_key: str, slot: str) -> str:
    """Look up the human-reviewed template sentence. Empty slot -> flagged
    placeholder (mockup only; production must not show placeholders)."""
    templates = json.loads(TEMPLATES_PATH.read_text())
    sentence = templates.get(module_key, {}).get(criterion_key, {}).get(slot, "")
    if sentence:
        return sentence
    return (f"[PLACEHOLDER — template '{module_key}.{criterion_key}.{slot}' "
            f"not yet written/reviewed. Real sentence to be authored and "
            f"human-reviewed before production use.]")


def _modifier_heading_phrase(points: int) -> str:
    """Plain-language heading phrase for modifier-style terms (e.g. the
    McIsaac age adjustment), which are not simple present/absent criteria
    and should not show a bare signed integer ("-1 point") as their
    primary heading text. Regular present/absent criteria keep their
    existing "+1 point" heading format (see the CENTOR_BOOL_CRITERIA loop
    above) since those aren't ambiguous the same way.

    The raw point value is unaffected elsewhere: it's still passed through
    to generate_criterion_diagram() and used in the score total exactly
    as before.
    """
    if points > 0:
        return f"this raised your score by {points} point" + ("s" if points != 1 else "")
    elif points < 0:
        n = abs(points)
        return f"this lowered your score by {n} point" + ("s" if n != 1 else "")
    else:
        return "this did not change your score"


def _age_modifier_slot(age: int) -> str:
    if age < 18:
        raise ValueError(
            f"age {age} is below this tool's scope (adults >=18); "
            "age_3_to_14 McIsaac bracket is not supported"
        )
    elif age <= 44:
        return "age_15_to_44"
    else:
        return "age_45_plus"


def build_mockup(scenario: dict) -> None:
    sample_criteria = scenario["criteria"]
    sample_patient_words = scenario["patient_words"]
    age = scenario["age"]

    # 1. Real rule-engine scoring output
    result = calculate_centor_score(age=age, **sample_criteria)
    score = result["score"]
    tier = result["tier"]

    # 2. Published probability (cited-only; raises if bracket uncited)
    try:
        prob = get_centor_probability(score)
    except UncitedScoreError as e:
        print(f"REFUSING to show a probability: {e}", file=sys.stderr)
        sys.exit(1)

    md = []
    md.append("# Your Sore Throat Assessment — How Your Score Was Calculated\n")
    md.append(f"> {EXPLANATION_DISCLAIMER}\n")
    md.append(f"**Your Modified Centor (McIsaac) score: {score}** "
              f"({tier} probability tier, {prob.score_bracket})\n")
    md.append(f"**What this range means (published data):** In validation "
              f"studies, {prob.probability_text} of people in this range "
              f"were found to have streptococcal pharyngitis (strep "
              f"throat).\n")
    md.append("**Probability estimate source:**")
    md.append(f"*{prob.citation}*\n")
    md.append("---\n")
    md.append("## What contributed to your score\n")

    # Boolean criteria (fever, absence_of_cough, tender_cervical_nodes, tonsillar_exudate)
    for param_key, label, template_key in CENTOR_BOOL_CRITERIA:
        present = sample_criteria[param_key]
        points = result["breakdown"][label]["points"]
        if present:
            sentence = load_template("modified_centor", template_key, "present")
            patient_words = sample_patient_words.get(param_key, "(from your answers)")
            md.append(f"### {label} — +{points} point\n")
            md.append(f"- **You told us:** \u201c{patient_words}\u201d")
            md.append(f"- **Explanation:** {sentence}\n")
            svg = generate_criterion_diagram(
                patient_words=patient_words,
                criterion_label=label,
                criterion_matched=True,
                points_contributed=points,
                total_score=score,
                tier=tier,
            )
            svg_name = f"centor_{param_key}.svg"
            (OUT_DIR / svg_name).write_text(svg)
            md.append(f"![How {label} affected your score]({svg_name})\n")

    # Age modifier: always applies (not a present/absent boolean), so it's
    # shown unconditionally rather than gated on `if present`, using
    # match_text_override since "matched"/"not matched" doesn't fit a
    # continuous/modifier-style term.
    age_points = result["breakdown"]["Age"]["points"]
    age_slot = _age_modifier_slot(age)
    age_sentence = load_template("modified_centor", "age_modifier", age_slot)
    md.append(f"### Age adjustment (age {age}) — {_modifier_heading_phrase(age_points)}\n")
    md.append(f"- **You told us:** \u201cI'm {age} years old\u201d")
    md.append(f"- **Explanation:** {age_sentence}\n")
    age_svg = generate_criterion_diagram(
        patient_words=f"I'm {age} years old",
        criterion_label="Age adjustment",
        criterion_matched=(age_points != 0),
        points_contributed=age_points,
        total_score=score,
        tier=tier,
        match_text_override=(
            f"McIsaac age modifier: {age_points:+d} point"
            + ("s" if age_points not in (1, -1) else "")
        ),
    )
    age_svg_name = "centor_age_modifier.svg"
    (OUT_DIR / age_svg_name).write_text(age_svg)
    md.append(f"![How age affected your score]({age_svg_name})\n")

    md.append("## What did NOT contribute\n")
    absent = [label for param_key, label, _ in CENTOR_BOOL_CRITERIA if not sample_criteria[param_key]]
    if absent:
        md.append("These criteria were checked and not matched (0 points each): "
                  + ", ".join(absent) + ".\n")
    else:
        md.append("All four Centor criteria were matched.\n")

    md.append("## Pending clinical evaluation\n")
    md.append("This is a PARTIAL score. The following require in-person "
              "assessment and are not included: "
              + ", ".join(result["pendingFields"]) + ".\n")

    md.append("---\n")
    md.append(f"> {EXPLANATION_DISCLAIMER}\n")
    md.append("**Scoring criteria source:**")
    md.append(f"*{result['citation']}*\n")

    out_md = OUT_DIR / scenario["output_filename"]
    out_md.write_text("\n".join(md))

    print(f"[{scenario['name']}]")
    print(f"  Mockup written to: {out_md}")
    print(f"  Sample data: score={score}, tier={tier}, "
          f"published probability={prob.probability_text}")
    print()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        build_mockup(scenario)


if __name__ == "__main__":
    main()
