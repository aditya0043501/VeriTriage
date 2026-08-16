"""
Centor Criteria (McIsaac modification) - Deterministic clinical scoring function
Sources:
- Centor RM, et al. "The diagnosis of strep throat in adults in the emergency
  room." Med Decis Making. 1981;1(3):239-46.
- McIsaac WJ, et al. "A clinical score to reduce unnecessary antibiotic use in
  patients with sore throat." CMAJ. 1998;158(1):75-83.

Criteria (1 point each, patient-knowable):
- Fever (temperature > 38 C / 100.4 F, measured or reported feverish)
- Absence of cough
- Tender/swollen lymph nodes at the front of the neck (anterior cervical)
- Tonsillar exudate or swelling (white patches / swollen tonsils)

McIsaac age modification (published):
- Age 3-14: +1 point (never applies here; tool scope is adults >=18)
- Age 15-44: 0 points
- Age >=45: -1 point

Clinical-test-only (marked pending): rapid strep test, throat culture.

Risk stratification (McIsaac validated probabilities of strep):
- Score <=1: Low (~1-10%)
- Score 2-3: Moderate (~11-35%)
- Score >=4: High (~51%+)
"""

CITATION = ("Centor RM, et al. Med Decis Making. 1981;1(3):239-46; "
            "McIsaac WJ, et al. CMAJ. 1998;158(1):75-83.")
CITATION_URL = "https://pubmed.ncbi.nlm.nih.gov/6763125/"
CITATION_DOI = "https://doi.org/10.1177/0272989X8100100304"
CITATION_URL_2 = "https://pubmed.ncbi.nlm.nih.gov/9475915/"


def calculate_centor_score(
    fever: bool,
    absence_of_cough: bool,
    tender_cervical_nodes: bool,
    tonsillar_exudate: bool,
    age: int
) -> dict:
    """
    Calculate the McIsaac-modified Centor score for sore throat.
    """
    if age < 0:
        raise ValueError(f"age must be non-negative, got {age}")

    breakdown = {}
    total_score = 0

    fever_pts = 1 if fever else 0
    total_score += fever_pts
    breakdown["Fever"] = {
        "points": fever_pts,
        "justification": ("Fever over 38\u00b0C contributes 1 point per Centor criteria" if fever
                          else "No fever reported: 0 points per Centor criteria")
    }

    cough_pts = 1 if absence_of_cough else 0
    total_score += cough_pts
    breakdown["Absence of cough"] = {
        "points": cough_pts,
        "justification": ("Absence of cough contributes 1 point per Centor criteria (bacterial throat infections typically lack cough)" if absence_of_cough
                          else "Cough present: 0 points per Centor criteria")
    }

    nodes_pts = 1 if tender_cervical_nodes else 0
    total_score += nodes_pts
    breakdown["Tender neck lymph nodes"] = {
        "points": nodes_pts,
        "justification": ("Tender lymph nodes at the front of the neck contribute 1 point per Centor criteria" if tender_cervical_nodes
                          else "No tender neck lymph nodes reported: 0 points per Centor criteria")
    }

    exudate_pts = 1 if tonsillar_exudate else 0
    total_score += exudate_pts
    breakdown["Tonsillar exudate/swelling"] = {
        "points": exudate_pts,
        "justification": ("White patches or swelling on the tonsils contribute 1 point per Centor criteria" if tonsillar_exudate
                          else "No tonsillar white patches or swelling reported: 0 points per Centor criteria")
    }

    # McIsaac age modification (published values)
    if age <= 14:
        age_pts = 1
        age_just = f"Age {age}: +1 point per McIsaac modification (strep is more common in children)"
    elif age <= 44:
        age_pts = 0
        age_just = f"Age {age}: 0 points per McIsaac modification (ages 15-44)"
    else:
        age_pts = -1
        age_just = f"Age {age}: -1 point per McIsaac modification (strep is less common at age 45 and over)"
    total_score += age_pts
    breakdown["Age"] = {"points": age_pts, "justification": age_just}

    pending_fields = ["Rapid strep test / throat culture"]
    breakdown["Rapid strep test / throat culture"] = {
        "points": "pending clinical evaluation",
        "justification": "Confirming a strep infection requires a rapid strep test or throat culture (requires in-person testing)"
    }
    is_partial = True

    # Fixed published cutoffs (McIsaac three-band consolidation)
    if total_score <= 1:
        tier = "low"
        what_to_do = "Self-care is reasonable: rest, fluids, warm liquids, and standard over-the-counter pain relief if you normally use it. If symptoms worsen, you develop trouble breathing or swallowing, or a fever persists beyond 2-3 days, seek care."
        who_to_see = "Primary care physician if symptoms persist"
        how_soon = "Routine follow-up is fine; sooner if symptoms worsen"
        full_text = "LOW LIKELIHOOD (score <=1): Roughly 1-10% chance of strep throat. Most sore throats in this range are viral and get better on their own. This is an estimate based on your reported symptoms, not a diagnosis."
    elif total_score <= 3:
        tier = "moderate"
        what_to_do = "This needs clinical follow-up for a strep test. No specific home treatment replaces testing at this level."
        who_to_see = "Primary care physician or urgent care"
        how_soon = "Within 24 hours"
        full_text = "MODERATE LIKELIHOOD (score 2-3): Roughly 11-35% chance of strep throat. A quick in-person strep test within 24 hours is recommended to decide whether antibiotics are needed. This is an estimate based on your reported symptoms, not a diagnosis."
    else:
        tier = "high"
        what_to_do = "Seek in-person care for testing and likely treatment. No home remedies replace evaluation at this level."
        who_to_see = "Primary care physician or urgent care"
        how_soon = "Within the next few hours, today"
        full_text = "HIGH LIKELIHOOD (score >=4): Roughly a 50% or higher chance of strep throat. Prompt in-person testing and treatment is recommended today. This is an estimate based on your reported symptoms, not a diagnosis."

    full_text += " NOTE: This is a PARTIAL score based on what you can report at home. Confirming strep requires an in-person rapid strep test or throat culture."

    return {
        "score": total_score,
        "isPartial": is_partial,
        "pendingFields": pending_fields,
        "tier": tier,
        "breakdown": breakdown,
        "citation": CITATION,
        "citation_url": CITATION_URL,
        "citation_doi": CITATION_DOI,
        "citation_url_2": CITATION_URL_2,
        "recommendation": {
            "what_to_do": what_to_do,
            "who_to_see": who_to_see,
            "how_soon": how_soon,
            "full_text": full_text
        }
    }
