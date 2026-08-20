#!/usr/bin/env python3
"""
Stage 4 worked mockup: Wells' DVT score explanation walkthrough.

Standalone demo — NOT wired into main.py or the production pipeline.
Runs the ACTUAL scoring engine (calculate_wells_score) on real sample data,
then assembles the patient-facing explanation from:
  - explanations/templates.json  (human-reviewed sentences; empty slots fall
    back to clearly-flagged [PLACEHOLDER] text for this mockup ONLY)
  - probability_mapping.py       (published, cited probability ranges only)
  - diagram_generator.py         (deterministic SVG, from rule-engine output)
  - disclaimer.py                (fixed disclaimer text)

Two citations appear in each mockup, for two different facts, and are
labeled accordingly so they don't read as inconsistent:
  - "Probability estimate source" -> Wells 2006 JAMA (pooled DVT prevalence
    by clinical probability tier; used by probability_mapping.py)
  - "Scoring criteria source" -> Wells 2003 NEJM (the citation already
    returned by calculate_wells_score(), i.e. the source for the point
    criteria themselves, not the probability numbers)

Each matched criterion is shown with exactly ONE rendering of its diagram
(an embedded image), not also a separate plain-text filename bullet — the
embed already conveys the file and renders it inline.

Output: one markdown file + one SVG per matched criterion per scenario,
written to backend/explanations/mockup_output/.
"""

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = THIS_DIR.parent
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scoring.wells_score import calculate_wells_score, CRITERIA  # noqa: E402
from explanations.probability_mapping import get_wells_probability, UncitedScoreError  # noqa: E402
from explanations.diagram_generator import generate_criterion_diagram  # noqa: E402
from explanations.disclaimer import EXPLANATION_DISCLAIMER  # noqa: E402

TEMPLATES_PATH = REPO_ROOT / "explanations" / "templates.json"
OUT_DIR = THIS_DIR / "mockup_output"

# ---------------------------------------------------------------------------
# Scenario 1: a plausible, intuitive patient conversation outcome.
# "My left calf is really sore when I press it, and it looks noticeably
#  bigger than the right one."
# Extraction yields: localized_tenderness=True, calf_swelling_over_3cm=True,
# everything else False. Wells score = 2 (moderate tier).
# ---------------------------------------------------------------------------
SCENARIO_TENDERNESS_CALF = {
    "name": "Tenderness + calf swelling (intuitive criteria)",
    "output_filename": "explanations_mockup_wells.md",
    "patient_words": {
        "localized_tenderness": "it's really sore when I press on my calf",
        "calf_swelling_over_3cm": "my left calf looks noticeably bigger than the right",
    },
    "criteria": {
        "active_cancer": False,
        "paralysis_or_immobilization": False,
        "bedridden_or_surgery": False,
        "localized_tenderness": True,
        "entire_leg_swollen": False,
        "calf_swelling_over_3cm": True,
        "pitting_edema": False,
        "collateral_veins": False,
    },
}

# ---------------------------------------------------------------------------
# Scenario 2: a HARDER-to-explain criterion — "Collateral superficial veins
# (non-varicose)". Unlike tenderness or swelling, this criterion requires
# explaining an unfamiliar clinical concept (the body rerouting blood flow
# through surface veins when a deep vein is blocked) in plain language,
# which stress-tests whether the template slot structure holds up.
# "I've noticed some new veins showing on the surface of my leg that
#  weren't there before."
# Extraction yields: collateral_veins=True, everything else False.
# Wells score = 1 (moderate tier).
# ---------------------------------------------------------------------------
SCENARIO_COLLATERAL_VEINS = {
    "name": "Collateral superficial veins (harder-to-explain criterion)",
    "output_filename": "explanations_mockup_wells_collateral_veins.md",
    "patient_words": {
        "collateral_veins": "I've noticed some new veins showing on the surface of my leg that weren't there before",
    },
    "criteria": {
        "active_cancer": False,
        "paralysis_or_immobilization": False,
        "bedridden_or_surgery": False,
        "localized_tenderness": False,
        "entire_leg_swollen": False,
        "calf_swelling_over_3cm": False,
        "pitting_edema": False,
        "collateral_veins": True,
    },
}

SCENARIOS = [SCENARIO_TENDERNESS_CALF, SCENARIO_COLLATERAL_VEINS]


def load_template(module_key: str, criterion_key: str, present: bool) -> str:
    """Look up the human-reviewed template sentence. Empty slot -> flagged
    placeholder (mockup only; production must not show placeholders)."""
    templates = json.loads(TEMPLATES_PATH.read_text())
    slot = "present" if present else "absent"
    sentence = templates.get(module_key, {}).get(criterion_key, {}).get(slot, "")
    if sentence:
        return sentence
    return (f"[PLACEHOLDER — template '{module_key}.{criterion_key}.{slot}' "
            f"not yet written/reviewed. Real sentence to be authored and "
            f"human-reviewed before production use.]")


def build_mockup(scenario: dict) -> None:
    sample_criteria = scenario["criteria"]
    sample_patient_words = scenario["patient_words"]

    # 1. Real rule-engine scoring output
    result = calculate_wells_score(**sample_criteria)
    score = result["score"]
    tier = result["tier"]

    # 2. Published probability (cited-only; raises if bracket uncited)
    try:
        prob = get_wells_probability(score)
    except UncitedScoreError as e:
        print(f"REFUSING to show a probability: {e}", file=sys.stderr)
        sys.exit(1)

    md = []
    md.append("# Your Leg Swelling Assessment — How Your Score Was Calculated\n")
    md.append(f"> {EXPLANATION_DISCLAIMER}\n")
    md.append(f"**Your Wells' DVT score: {score}** "
              f"({tier} probability tier, {prob.score_bracket})\n")
    md.append(f"**What this range means (published data):** In pooled studies, "
              f"{prob.probability_text} of people in this range were found to "
              f"have a DVT.\n")
    md.append("**Probability estimate source:**")
    md.append(f"*{prob.citation}*\n")
    md.append("---\n")
    md.append("## What contributed to your score\n")

    for key, label, _justification in CRITERIA:
        present = sample_criteria[key]
        points = result["breakdown"][label]["points"]
        if present:
            sentence = load_template("wells_dvt", key, present=True)
            patient_words = sample_patient_words.get(key, "(from your answers)")
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
            svg_name = f"wells_{key}.svg"
            (OUT_DIR / svg_name).write_text(svg)
            # Single rendering of the diagram: an embedded image. No separate
            # plain-text "Diagram: <filename>" bullet — that was redundant
            # with the embed, which already names and renders the file.
            md.append(f"![How {label} affected your score]({svg_name})\n")

    md.append("## What did NOT contribute\n")
    absent = [label for key, label, _ in CRITERIA if not sample_criteria[key]]
    md.append("These criteria were checked and not matched (0 points each): "
              + ", ".join(absent) + ".\n")

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
