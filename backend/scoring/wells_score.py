"""
Wells' Criteria for DVT - Deterministic clinical scoring function
Source: Wells PS, et al. "Evaluation of D-dimer in the diagnosis of suspected
deep-vein thrombosis." N Engl J Med. 2003;349(13):1227-35.

Criteria (1 point each, patient-knowable):
- Active cancer (treatment ongoing or within 6 months)
- Paralysis, paresis, or recent plaster immobilization of lower extremities
- Recently bedridden >3 days or major surgery within 12 weeks
- Localized tenderness along the deep venous system
- Entire leg swollen
- Calf swelling >3 cm compared to asymptomatic side
- Pitting edema confined to the symptomatic leg
- Collateral superficial veins (non-varicose)

Clinical-evaluation-only (marked pending, requires in-person assessment):
- Alternative diagnosis at least as likely as DVT (-2 points, clinician judgment)
- D-dimer / ultrasound (confirmatory testing, not part of the point score)

Risk stratification (published cutoffs):
- Score <=0: Low probability (~5% DVT prevalence)
- Score 1-2: Moderate probability (~17% DVT prevalence)
- Score >=3: High probability (~17-53% DVT prevalence)
"""

CRITERIA = [
    ("active_cancer", "Active cancer", "Active cancer (treatment ongoing or within 6 months) contributes 1 point per Wells' criteria"),
    ("paralysis_or_immobilization", "Paralysis / immobilization", "Paralysis, paresis, or recent leg immobilization contributes 1 point per Wells' criteria"),
    ("bedridden_or_surgery", "Bedridden / recent surgery", "Recently bedridden >3 days or major surgery within 12 weeks contributes 1 point per Wells' criteria"),
    ("localized_tenderness", "Localized tenderness", "Tenderness along the deep vein system contributes 1 point per Wells' criteria"),
    ("entire_leg_swollen", "Entire leg swollen", "Swelling of the entire leg contributes 1 point per Wells' criteria"),
    ("calf_swelling_over_3cm", "Calf swelling >3 cm", "Calf swelling >3 cm versus the other leg contributes 1 point per Wells' criteria"),
    ("pitting_edema", "Pitting edema", "Pitting edema confined to the symptomatic leg contributes 1 point per Wells' criteria"),
    ("collateral_veins", "Collateral superficial veins", "Non-varicose collateral superficial veins contribute 1 point per Wells' criteria"),
]

CITATION = "Wells PS, et al. 'Evaluation of D-dimer in the diagnosis of suspected deep-vein thrombosis.' N Engl J Med. 2003;349(13):1227-35."
CITATION_URL = "https://pubmed.ncbi.nlm.nih.gov/14507948/"
CITATION_DOI = "https://doi.org/10.1056/nejmoa023153"


def calculate_wells_score(**criteria) -> dict:
    """
    Calculate Wells' DVT score from boolean criteria.

    Keyword args: one bool per criterion key in CRITERIA (missing = False).
    The 'alternative diagnosis' adjustment (-2) requires clinician judgment and
    is always marked pending; the returned score is therefore a partial score.
    """
    breakdown = {}
    total_score = 0

    for key, label, justification in CRITERIA:
        present = bool(criteria.get(key, False))
        points = 1 if present else 0
        total_score += points
        breakdown[label] = {
            "points": points,
            "justification": justification if present else f"{label}: 0 points (not reported) per Wells' criteria"
        }

    # Clinician-judgment / clinical-test-only fields - always pending here
    pending_fields = ["Alternative diagnosis assessment", "D-dimer / ultrasound"]
    breakdown["Alternative diagnosis assessment"] = {
        "points": "pending clinical evaluation",
        "justification": "Whether another diagnosis is at least as likely (-2 points) requires clinician judgment (requires in-person evaluation)"
    }
    breakdown["D-dimer / ultrasound"] = {
        "points": "pending clinical evaluation",
        "justification": "D-dimer blood test and ultrasound confirmation require in-person testing"
    }
    is_partial = True

    # Fixed published cutoffs
    if total_score <= 0:
        tier = "low"
        what_to_do = "Self-monitor the leg. Keep it elevated when resting and stay mobile. If swelling, pain, or redness worsens, or you develop shortness of breath or chest pain, seek care immediately."
        who_to_see = "Primary care physician"
        how_soon = "Routine follow-up within the next few days"
        full_text = "LOW PROBABILITY (score <=0): About 5% of people in this range have a DVT. Most do not, but a doctor can order a simple blood test (D-dimer) to be sure. This is an estimate based on your reported symptoms, not a diagnosis."
    elif total_score <= 2:
        tier = "moderate"
        what_to_do = "This needs clinical follow-up. No home remedies for this risk level. If you develop shortness of breath or chest pain, seek emergency care immediately."
        who_to_see = "Primary care physician, urgent care, or vascular specialist"
        how_soon = "Within 24 hours"
        full_text = "MODERATE PROBABILITY (score 1-2): About 17% of people in this range have a DVT. A clinical evaluation within 24 hours, including blood testing, is recommended to be sure. This is an estimate based on your reported symptoms, not a diagnosis."
    else:
        tier = "high"
        what_to_do = "Seek immediate in-person care. No home remedies for this risk level. Do not massage the leg."
        who_to_see = "Emergency room, due to blood clot risk"
        how_soon = "Within the next few hours - go now"
        full_text = "HIGH PROBABILITY (score >=3): A substantial share of people in this range have a DVT, which can be dangerous if untreated. Immediate evaluation with ultrasound is recommended. This is an estimate based on your reported symptoms, not a diagnosis."

    full_text += " NOTE: This is a PARTIAL score based on what you can report at home. A complete evaluation requires clinician assessment and testing (alternative diagnosis review, D-dimer/ultrasound)."

    return {
        "score": total_score,
        "isPartial": is_partial,
        "pendingFields": pending_fields,
        "tier": tier,
        "breakdown": breakdown,
        "citation": CITATION,
        "citation_url": CITATION_URL,
        "citation_doi": CITATION_DOI,
        "recommendation": {
            "what_to_do": what_to_do,
            "who_to_see": who_to_see,
            "how_soon": how_soon,
            "full_text": full_text
        }
    }
