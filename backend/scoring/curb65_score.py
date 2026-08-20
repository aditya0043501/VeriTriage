"""
CURB-65 score for community-acquired pneumonia severity - deterministic
clinical scoring function.

Source: Lim WS, van der Eerden MM, Laing R, et al. "Defining community
acquired pneumonia severity on presentation to hospital: an international
derivation and validation study." Thorax. 2003;58(5):377-382.

Criteria (1 point each):
- C: Confusion (new onset)
- U: Urea >7 mmol/L (or BUN >20 mg/dL)
- R: Respiratory rate >=30 breaths/min
- B: Blood pressure low (systolic <90 or diastolic <=60 mmHg)
- 65: Age >=65 years

Risk stratification (published 30-day mortality, Lim 2003 derivation):
- Score 0:   0.6%  — low
- Score 1:   2.7%  — low
- Score 2:   6.8%  — moderate
- Score 3:  14.0%  — severe
- Score 4:  27.8%  — severe
- Score 5:  27.8%  — severe

Patient-reportable caveat: urea and blood pressure are measurements, not
things every patient knows. Patients who answer "no" or "not sure" are
scored as not-elevated and the report flags that those tests are pending,
so the score is a floor, not a ceiling.
"""

CRITERIA = [
    ("confusion", "New confusion", "New-onset confusion or disorientation contributes 1 point per CURB-65 criteria"),
    ("urea_elevated", "High urea / BUN", "Urea >7 mmol/L (BUN >20 mg/dL) on blood test contributes 1 point per CURB-65 criteria"),
    ("rr_high", "Fast breathing", "Respiratory rate >=30 breaths/min contributes 1 point per CURB-65 criteria"),
    ("bp_low", "Low blood pressure", "Systolic <90 or diastolic <=60 mmHg contributes 1 point per CURB-65 criteria"),
    ("age_65_plus", "Age 65 or older", "Age >=65 years contributes 1 point per CURB-65 criteria"),
]

MORTALITY_BY_SCORE = {
    0: "0.6%",
    1: "2.7%",
    2: "6.8%",
    3: "14.0%",
    4: "27.8%",
    5: "27.8%",
}

CITATION = ("Lim WS, van der Eerden MM, Laing R, et al. 'Defining community acquired "
            "pneumonia severity on presentation to hospital: an international derivation "
            "and validation study.' Thorax. 2003;58(5):377-382.")
CITATION_URL = "https://pubmed.ncbi.nlm.nih.gov/12728155/"
CITATION_DOI = "https://doi.org/10.1136/thorax.58.5.377"


def calculate_curb65_score(
    confusion: bool,
    urea_elevated: bool,
    rr_high: bool,
    bp_low: bool,
    age_65_plus: bool,
) -> dict:
    """Calculate the CURB-65 score from boolean criteria."""
    breakdown = {}
    total_score = 0

    inputs = {
        "confusion": confusion,
        "urea_elevated": urea_elevated,
        "rr_high": rr_high,
        "bp_low": bp_low,
        "age_65_plus": age_65_plus,
    }
    for key, label, justification in CRITERIA:
        present = bool(inputs[key])
        points = 1 if present else 0
        total_score += points
        breakdown[label] = {
            "points": points,
            "justification": justification if present else f"{label}: 0 points (not reported) per CURB-65 criteria"
        }

    pending_fields = ["Chest X-ray confirmation", "Blood urea / BUN test (if not yet done)"]
    breakdown["Chest X-ray confirmation"] = {
        "points": "pending clinical evaluation",
        "justification": "Pneumonia diagnosis is confirmed with a chest X-ray (requires in-person evaluation)"
    }
    breakdown["Blood urea / BUN test (if not yet done)"] = {
        "points": "pending clinical evaluation",
        "justification": "Blood urea is a lab measurement; if not yet tested it is scored as not-elevated (score is a floor, not a ceiling)"
    }
    is_partial = True

    mortality = MORTALITY_BY_SCORE[total_score]

    if total_score <= 1:
        tier = "low"
        what_to_do = "Home care with close monitoring is usually appropriate at this level. Contact your doctor if symptoms worsen — especially increasing breathlessness, confusion, or a fever that persists."
        who_to_see = "Primary care physician"
        how_soon = "Routine follow-up within 1-2 days"
        full_text = (f"LOW RISK (score {total_score}): In the CURB-65 derivation study, 30-day mortality in "
                     f"this range was {mortality}. Outpatient or home care with monitoring is usually "
                     f"appropriate. This is an estimate based on your reported symptoms, not a diagnosis.")
    elif total_score == 2:
        tier = "moderate"
        what_to_do = "This needs clinical evaluation promptly. A short hospital stay may be recommended. If you become more breathless or confused, seek emergency care."
        who_to_see = "Primary care physician or urgent care"
        how_soon = "Same day"
        full_text = (f"MODERATE RISK (score 2): In the CURB-65 derivation study, 30-day mortality in this "
                     f"range was {mortality}. Prompt same-day clinical evaluation is recommended; a short "
                     f"hospital stay may be advised. This is an estimate based on your reported symptoms, "
                     f"not a diagnosis.")
    else:
        tier = "severe"
        what_to_do = "Seek emergency care now. Hospital admission — possibly intensive care — is recommended at this level."
        who_to_see = "Emergency department"
        how_soon = "Immediately — go now"
        full_text = (f"SEVERE RISK (score {total_score}): In the CURB-65 derivation study, 30-day mortality in "
                     f"this range was {mortality}. Hospital admission is recommended, with urgent assessment "
                     f"for intensive care. This is an estimate based on your reported symptoms, not a diagnosis.")

    full_text += (" NOTE: This is a PARTIAL score based on what you can report at home. Urea/BUN and blood "
                  "pressure are measurements; if you haven't had them tested, they are scored as normal, "
                  "which means your score could be higher after clinical testing.")

    return {
        "score": total_score,
        "isPartial": is_partial,
        "pendingFields": pending_fields,
        "tier": tier,
        "mortality_30_day": mortality,
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
