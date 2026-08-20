"""
Explanation builder — assembles the patient-facing explanation layer
from the same rule-engine output already used for scoring.

This module wires together:
  - explanations/templates.json  (human-reviewed sentences; empty slots
    cause the explanation to be unavailable — NO placeholders in production)
  - probability_mapping.py       (published, cited probability ranges only)
  - diagram_generator.py         (deterministic SVG, from rule-engine output)
  - disclaimer.py                (fixed disclaimer text)

Returns a structured dict suitable for inclusion in the /api/chat
doctor_report response as an additive `explanation` field. Does NOT
modify or replace any existing clinical-grade output.

Stage 7 scope: Wells' DVT and Modified Centor only. Returns None for
CHA₂DS₂-VASc (not wired in this stage).
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

from explanations.probability_mapping import (
    get_wells_probability,
    get_centor_probability,
    UncitedScoreError,
)
from explanations.diagram_generator import generate_criterion_diagram
from explanations.disclaimer import EXPLANATION_DISCLAIMER

TEMPLATES_PATH = Path(__file__).resolve().parent.parent.parent / "explanations" / "templates.json"

_QUOTE_MAX_LEN = 80


def _truncate_quote(quote: str) -> str:
    """Cap a patient quote at 80 characters, with an ellipsis if longer."""
    quote = quote.strip()
    if len(quote) <= _QUOTE_MAX_LEN:
        return quote
    return quote[:_QUOTE_MAX_LEN].rstrip() + "..."


def _patient_words(param_key: str, source_quotes: Dict[str, str]) -> Optional[str]:
    """The patient's own words for a matched criterion, or None when no
    quote was captured (the frontend then shows 'Based on your
    description' — never a templated placeholder)."""
    quote = source_quotes.get(param_key)
    return _truncate_quote(quote) if quote else None

# Wells' criteria: (param_key, breakdown_label, template_key)
_WELLS_CRITERIA = [
    ("active_cancer", "Active cancer", "active_cancer"),
    ("paralysis_or_immobilization", "Paralysis / immobilization", "paralysis_or_immobilization"),
    ("bedridden_or_surgery", "Bedridden / recent surgery", "bedridden_or_surgery"),
    ("localized_tenderness", "Localized tenderness", "localized_tenderness"),
    ("entire_leg_swollen", "Entire leg swollen", "entire_leg_swollen"),
    ("calf_swelling_over_3cm", "Calf swelling >3 cm", "calf_swelling_over_3cm"),
    ("pitting_edema", "Pitting edema", "pitting_edema"),
    ("collateral_veins", "Collateral superficial veins", "collateral_veins"),
]

# Centor boolean criteria: (param_key, breakdown_label, template_key)
_CENTOR_BOOL_CRITERIA = [
    ("fever", "Fever", "fever"),
    ("absence_of_cough", "Absence of cough", "absence_of_cough"),
    ("tender_cervical_nodes", "Tender neck lymph nodes", "tender_cervical_nodes"),
    ("tonsillar_exudate", "Tonsillar exudate/swelling", "tonsillar_exudate"),
]


def _load_templates() -> dict:
    return json.loads(TEMPLATES_PATH.read_text())


def _get_template(templates: dict, module_key: str, criterion_key: str, slot: str) -> str:
    """Look up a human-reviewed template sentence. Returns empty string
    if the slot is empty (caller must check and treat as unavailable)."""
    return templates.get(module_key, {}).get(criterion_key, {}).get(slot, "")


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


def _modifier_heading_phrase(points: int) -> str:
    if points > 0:
        return f"this raised your score by {points} point" + ("s" if points != 1 else "")
    elif points < 0:
        n = abs(points)
        return f"this lowered your score by {n} point" + ("s" if n != 1 else "")
    else:
        return "this did not change your score"


def _build_wells_explanation(
    score_result: dict,
    scoring_variables: dict,
    templates: dict,
    source_quotes: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    """Build the explanation dict for Wells' DVT. Returns None if any
    needed template slot is empty (no placeholders in production)."""
    score = score_result["score"]
    tier = score_result["tier"]
    breakdown = score_result["breakdown"]

    # Published probability (cited-only)
    try:
        prob = get_wells_probability(score)
    except UncitedScoreError:
        return None

    criteria_list: List[dict] = []
    not_contributed: List[str] = []

    for param_key, label, template_key in _WELLS_CRITERIA:
        present = bool(scoring_variables.get(param_key, False))
        points = breakdown.get(label, {}).get("points", 0)
        slot = "present" if present else "absent"
        sentence = _get_template(templates, "wells_dvt", template_key, slot)
        if not sentence:
            return None  # Empty slot — explanation unavailable

        if present:
            quote = _patient_words(param_key, source_quotes or {})
            svg = generate_criterion_diagram(
                patient_words=quote or "Based on your description",
                criterion_label=label,
                criterion_matched=True,
                points_contributed=points,
                total_score=score,
                tier=tier,
            )
            criteria_list.append({
                "label": label,
                "matched": True,
                "points": points,
                "patient_words": quote,
                "explanation": sentence,
                "svg": svg,
            })
        else:
            not_contributed.append(label)

    return {
        "available": True,
        "disclaimer": EXPLANATION_DISCLAIMER,
        "score": score,
        "tier": tier,
        "score_bracket": prob.score_bracket,
        "probability_text": prob.probability_text,
        "probability_citation": prob.citation,
        "scoring_citation": score_result.get("citation", ""),
        "criteria": criteria_list,
        "not_contributed": not_contributed,
        "pending_fields": score_result.get("pendingFields", []),
    }


def _build_centor_explanation(
    score_result: dict,
    scoring_variables: dict,
    templates: dict,
    source_quotes: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    """Build the explanation dict for Modified Centor. Returns None if any
    needed template slot is empty (no placeholders in production)."""
    score = score_result["score"]
    tier = score_result["tier"]
    breakdown = score_result["breakdown"]
    age = scoring_variables.get("age")

    if age is None or not isinstance(age, int):
        return None

    # Published probability (cited-only)
    try:
        prob = get_centor_probability(score)
    except UncitedScoreError:
        return None

    criteria_list: List[dict] = []
    not_contributed: List[str] = []

    # Boolean criteria
    for param_key, label, template_key in _CENTOR_BOOL_CRITERIA:
        present = bool(scoring_variables.get(param_key, False))
        points = breakdown.get(label, {}).get("points", 0)
        slot = "present" if present else "absent"
        sentence = _get_template(templates, "modified_centor", template_key, slot)
        if not sentence:
            return None

        if present:
            quote = _patient_words(param_key, source_quotes or {})
            svg = generate_criterion_diagram(
                patient_words=quote or "Based on your description",
                criterion_label=label,
                criterion_matched=True,
                points_contributed=points,
                total_score=score,
                tier=tier,
            )
            criteria_list.append({
                "label": label,
                "matched": True,
                "points": points,
                "patient_words": quote,
                "explanation": sentence,
                "svg": svg,
            })
        else:
            not_contributed.append(label)

    # Age modifier (always applies — not a present/absent boolean)
    age_points = breakdown.get("Age", {}).get("points", 0)
    try:
        age_slot = _age_modifier_slot(age)
    except ValueError:
        return None

    age_sentence = _get_template(templates, "modified_centor", "age_modifier", age_slot)
    if not age_sentence:
        return None

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
    criteria_list.append({
        "label": f"Age adjustment (age {age})",
        "matched": age_points != 0,
        "points": age_points,
        "patient_words": f"I'm {age} years old",
        "explanation": age_sentence,
        "svg": age_svg,
        "heading_phrase": _modifier_heading_phrase(age_points),
    })

    return {
        "available": True,
        "disclaimer": EXPLANATION_DISCLAIMER,
        "score": score,
        "tier": tier,
        "score_bracket": prob.score_bracket,
        "probability_text": prob.probability_text,
        "probability_citation": prob.citation,
        "scoring_citation": score_result.get("citation", ""),
        "criteria": criteria_list,
        "not_contributed": not_contributed,
        "pending_fields": score_result.get("pendingFields", []),
    }


def build_explanation(
    category: str,
    score_result: dict,
    scoring_variables: dict,
    source_quotes: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    """Build the patient-facing explanation layer for a completed assessment.

    Returns a structured dict with template sentences, inline SVG diagrams,
    published probability, and disclaimer — or None if:
      - The category is afib_stroke (not wired in Stage 7)
      - Any needed template slot is empty (no placeholders in production)
      - The probability mapping raises UncitedScoreError

    This is ADDITIVE to the existing doctor_report — it does not modify
    or replace any clinical-grade output.
    """
    if category not in ("leg_swelling", "sore_throat"):
        return None

    try:
        templates = _load_templates()
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if category == "leg_swelling":
        return _build_wells_explanation(score_result, scoring_variables, templates, source_quotes)
    elif category == "sore_throat":
        return _build_centor_explanation(score_result, scoring_variables, templates, source_quotes)

    return None
