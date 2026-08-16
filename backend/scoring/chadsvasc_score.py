"""
CHA₂DS₂-VASc Score for Atrial Fibrillation Stroke Risk — Deterministic scoring function
Source: Lip GYH, et al. "Refining clinical risk stratification for predicting stroke
and thromboembolism in atrial fibrillation using a novel risk factor-based approach:
the euro heart survey on atrial fibrillation." Chest. 2010;137(2):263-72.

Criteria (verified against MDCalc and Merck Manual):
- Congestive heart failure history: 1 point
- Hypertension history: 1 point
- Age ≥ 75 years: 2 points
- Diabetes mellitus: 1 point
- Prior stroke / TIA / thromboembolism: 2 points
- Vascular disease (prior MI, peripheral artery disease, or aortic plaque): 1 point
- Age 65-74 years: 1 point
- Sex category (female): 1 point

Maximum score: 9 (female ≥75 with all risk factors)

All criteria are patient-observable — no lab or imaging dependency.
This is NOT a partial score; all inputs are self-reported by the patient.

Risk stratification (common clinical interpretation):
- Score 0: Low risk — anticoagulation not recommended
- Score 1: Moderate risk — anticoagulation may be considered
- Score ≥2: High risk — anticoagulation recommended
"""

CITATION = ("Lip GYH, et al. 'Refining clinical risk stratification for predicting stroke "
            "and thromboembolism in atrial fibrillation using a novel risk factor-based approach: "
            "the euro heart survey on atrial fibrillation.' Chest. 2010;137(2):263-72.")
CITATION_URL = "https://pubmed.ncbi.nlm.nih.gov/19762550/"
CITATION_DOI = "https://doi.org/10.1378/chest.09-1584"


def calculate_chadsvasc_score(
    age: int,
    sex: str,  # "male" or "female"
    chf_history: bool,
    hypertension: bool,
    stroke_tia_history: bool,
    vascular_disease: bool,
    diabetes: bool,
) -> dict:
    """
    Calculate CHA₂DS₂-VASc score from patient-reported criteria.

    All parameters are patient-observable (no clinical testing required).
    Returns a complete (non-partial) score with full breakdown.
    """
    breakdown = {}
    total_score = 0

    # C — Congestive heart failure (1 pt)
    pts = 1 if chf_history else 0
    total_score += pts
    breakdown["Congestive heart failure"] = {
        "points": pts,
        "justification": ("History of congestive heart failure contributes 1 point per CHA₂DS₂-VASc criteria"
                          if chf_history else "No heart failure history: 0 points per CHA₂DS₂-VASc criteria")
    }

    # H — Hypertension (1 pt)
    pts = 1 if hypertension else 0
    total_score += pts
    breakdown["Hypertension"] = {
        "points": pts,
        "justification": ("History of hypertension contributes 1 point per CHA₂DS₂-VASc criteria"
                          if hypertension else "No hypertension history: 0 points per CHA₂DS₂-VASc criteria")
    }

    # A₂ — Age ≥ 75 (2 pts)
    if age >= 75:
        pts = 2
        breakdown["Age ≥ 75"] = {
            "points": pts,
            "justification": f"Age {age} (≥75) contributes 2 points per CHA₂DS₂-VASc criteria"
        }
    elif age >= 65:
        pts = 1
        breakdown["Age 65-74"] = {
            "points": pts,
            "justification": f"Age {age} (65-74) contributes 1 point per CHA₂DS₂-VASc criteria"
        }
    else:
        pts = 0
        breakdown["Age < 65"] = {
            "points": pts,
            "justification": f"Age {age} (<65): 0 points per CHA₂DS₂-VASc criteria"
        }
    total_score += pts

    # D — Diabetes (1 pt)
    pts = 1 if diabetes else 0
    total_score += pts
    breakdown["Diabetes"] = {
        "points": pts,
        "justification": ("Diabetes mellitus contributes 1 point per CHA₂DS₂-VASc criteria"
                          if diabetes else "No diabetes: 0 points per CHA₂DS₂-VASc criteria")
    }

    # S₂ — Prior stroke/TIA/thromboembolism (2 pts)
    pts = 2 if stroke_tia_history else 0
    total_score += pts
    breakdown["Prior stroke/TIA"] = {
        "points": pts,
        "justification": ("Prior stroke, TIA, or thromboembolism contributes 2 points per CHA₂DS₂-VASc criteria"
                          if stroke_tia_history else "No prior stroke/TIA: 0 points per CHA₂DS₂-VASc criteria")
    }

    # V — Vascular disease (1 pt)
    pts = 1 if vascular_disease else 0
    total_score += pts
    breakdown["Vascular disease"] = {
        "points": pts,
        "justification": ("Vascular disease (prior MI, peripheral artery disease, or aortic plaque) "
                          "contributes 1 point per CHA₂DS₂-VASc criteria"
                          if vascular_disease else "No vascular disease: 0 points per CHA₂DS₂-VASc criteria")
    }

    # Sc — Sex category, female (1 pt)
    is_female = sex.lower().startswith("f")
    pts = 1 if is_female else 0
    total_score += pts
    breakdown["Sex (female)"] = {
        "points": pts,
        "justification": ("Female sex contributes 1 point per CHA₂DS₂-VASc criteria"
                          if is_female else "Male sex: 0 points per CHA₂DS₂-VASc criteria")
    }

    # Risk stratification
    if total_score == 0:
        tier = "low"
        what_to_do = ("Based on your score, current guidelines do not recommend anticoagulation. "
                      "This should be discussed with your doctor, who will consider your full clinical picture.")
        who_to_see = "Primary care physician or cardiologist"
        how_soon = "Routine follow-up"
        full_text = ("LOW RISK (score 0): Current guidelines suggest no anticoagulation is needed for "
                     "a CHA₂DS₂-VASc score of 0. This is an estimate based on your reported information, "
                     "not a diagnosis. Your doctor will consider your full clinical situation.")
    elif total_score == 1:
        tier = "moderate"
        what_to_do = ("Your score is in a borderline range where anticoagulation may be considered. "
                      "This decision should be made together with your doctor, weighing benefits and risks.")
        who_to_see = "Primary care physician or cardiologist"
        how_soon = "Within the next few weeks"
        full_text = ("MODERATE RISK (score 1): Anticoagulation may be considered for a CHA₂DS₂-VASc "
                     "score of 1. This is a decision to make with your doctor, who will weigh the "
                     "benefit of stroke prevention against bleeding risk. This is an estimate based "
                     "on your reported information, not a diagnosis.")
    else:
        tier = "high"
        what_to_do = ("Based on your score, current guidelines recommend anticoagulation to reduce "
                      "stroke risk. Please discuss this with your doctor as soon as feasible.")
        who_to_see = "Cardiologist or primary care physician"
        how_soon = "Within the next week"
        full_text = ("HIGH RISK (score ≥2): Current guidelines recommend anticoagulation for a "
                     "CHA₂DS₂-VASc score of 2 or higher. This significantly reduces stroke risk in "
                     "atrial fibrillation. Please discuss treatment options with your doctor. This is "
                     "an estimate based on your reported information, not a diagnosis.")

    return {
        "score": total_score,
        "isPartial": False,
        "pendingFields": [],
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
