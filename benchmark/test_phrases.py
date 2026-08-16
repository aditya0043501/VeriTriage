"""
153-phrase adversarial test set for VeriTriage extraction benchmark.

Each phrase has:
  - field: the scoring criterion being tested
  - phrase: the patient's words
  - expected: True (criterion present), False (criterion absent), or "unclear"
  - category: "yes" (affirmative), "no" (negative), or "hedged" (uncertain)
  - module: which scoring module the field belongs to
  - is_hard: whether this is a "hard" case (negation, hedging, or family-history)

This is the same test set used in the independent extraction audit.
"""

# Field descriptions for the LLM prompt — plain-language, no leading hints.
# These mirror what a patient would be asked, without revealing the scoring logic.
FIELD_DESCRIPTIONS = {
    # Centor / sore throat
    "fever": "The patient has a fever (temperature over 100.4°F / 38°C, or feels hot and shivery).",
    "absence_of_cough": "The patient does NOT have a cough. (Note: absence of cough is the positive criterion here — 'no cough' means this is True.)",
    "tender_cervical_nodes": "The patient has tender or swollen lymph nodes at the front of the neck, near the jaw.",
    "tonsillar_exudate": "The patient has white patches, pus, or swelling on their tonsils.",

    # Wells / leg swelling
    "active_cancer": "The patient is currently being treated for cancer, or has been treated for cancer in the last 6 months (chemo, radiation, tumor treatment).",
    "paralysis_or_immobilization": "The patient has paralysis, paresis, or recent immobilization of the leg (cast, splint, brace).",
    "bedridden_or_surgery": "The patient has been bedridden for more than 3 days recently, or had major surgery in the past 12 weeks.",
    "localized_tenderness": "The swollen leg is tender or painful along the deep vein system (inner calf or thigh) when pressed.",
    "entire_leg_swollen": "The entire leg is swollen (not just one part like the calf or ankle).",
    "calf_swelling_over_3cm": "The calf of the swollen leg is measurably bigger than the other side — more than 3 centimeters difference.",
    "pitting_edema": "The patient has pitting edema — pressing a finger into the swelling leaves a dent that stays for a moment.",
    "collateral_veins": "The patient has new visible surface veins on the swollen leg that weren't there before (collateral superficial veins).",

    # CHA₂DS₂-VASc / AFib
    "chf_history": "The patient has been diagnosed with heart failure (congestive heart failure / CHF).",
    "hypertension": "The patient has high blood pressure or has been told they have hypertension.",
    "stroke_tia_history": "The patient has had a prior stroke, mini-stroke (TIA), or thromboembolism.",
    "vascular_disease": "The patient has a personal history of vascular disease (heart attack, peripheral artery disease, aortic plaque, coronary artery disease, stent, or bypass).",
    "diabetes": "The patient has diabetes.",
}

# The 153 phrases: 17 fields × 3 categories (yes/no/hedged) × 3 phrases each
PHRASES = [
    # === fever (Centor) ===
    ("fever", "I've been running hot", True, "yes", "sore_throat", False),
    ("fever", "felt like I was burning up last night", True, "yes", "sore_throat", False),
    ("fever", "thermometer said 102", True, "yes", "sore_throat", False),
    ("fever", "temperature was normal", False, "no", "sore_throat", False),
    ("fever", "no fever at all", False, "no", "sore_throat", True),       # negation
    ("fever", "not hot or shivery", False, "no", "sore_throat", True),    # negation
    ("fever", "might have been a little warm", "unclear", "hedged", "sore_throat", True),
    ("fever", "hard to say", "unclear", "hedged", "sore_throat", True),
    ("fever", "maybe overnight", "unclear", "hedged", "sore_throat", True),

    # === absence_of_cough (Centor) — INVERTED POLARITY ===
    ("absence_of_cough", "no coughing", True, "yes", "sore_throat", True),   # negation = True (inverted)
    ("absence_of_cough", "I haven't coughed once", True, "yes", "sore_throat", True),
    ("absence_of_cough", "cough-free so far", True, "yes", "sore_throat", True),
    ("absence_of_cough", "yeah I keep coughing", False, "no", "sore_throat", False),
    ("absence_of_cough", "coughing up phlegm", False, "no", "sore_throat", False),
    ("absence_of_cough", "productive cough", False, "no", "sore_throat", False),
    ("absence_of_cough", "a tiny cough now and then", "unclear", "hedged", "sore_throat", True),
    ("absence_of_cough", "not much", "unclear", "hedged", "sore_throat", True),
    ("absence_of_cough", "occasionally", "unclear", "hedged", "sore_throat", True),

    # === tender_cervical_nodes (Centor) ===
    ("tender_cervical_nodes", "my neck glands hurt", True, "yes", "sore_throat", False),
    ("tender_cervical_nodes", "swollen lymph nodes under jaw", True, "yes", "sore_throat", False),
    ("tender_cervical_nodes", "sore lumps in throat area", True, "yes", "sore_throat", False),
    ("tender_cervical_nodes", "no neck lumps", False, "no", "sore_throat", True),
    ("tender_cervical_nodes", "glands feel normal", False, "no", "sore_throat", False),
    ("tender_cervical_nodes", "nothing swollen there", False, "no", "sore_throat", True),
    ("tender_cervical_nodes", "maybe a little tender", "unclear", "hedged", "sore_throat", True),
    ("tender_cervical_nodes", "I can't really tell", "unclear", "hedged", "sore_throat", True),
    ("tender_cervical_nodes", "slightly puffy perhaps", "unclear", "hedged", "sore_throat", True),

    # === tonsillar_exudate (Centor) ===
    ("tonsillar_exudate", "white dots on my tonsils", True, "yes", "sore_throat", False),
    ("tonsillar_exudate", "tonsils look spotty", True, "yes", "sore_throat", False),
    ("tonsillar_exudate", "patches of pus", True, "yes", "sore_throat", False),
    ("tonsillar_exudate", "tonsils are pink", False, "no", "sore_throat", False),
    ("tonsillar_exudate", "no spots back there", False, "no", "sore_throat", True),
    ("tonsillar_exudate", "looks clear", False, "no", "sore_throat", False),
    ("tonsillar_exudate", "hard to see back there", "unclear", "hedged", "sore_throat", True),
    ("tonsillar_exudate", "maybe some white", "unclear", "hedged", "sore_throat", True),
    ("tonsillar_exudate", "not sure if that's pus", "unclear", "hedged", "sore_throat", True),

    # === active_cancer (Wells) ===
    ("active_cancer", "I'm on chemo right now", True, "yes", "leg_swelling", False),
    ("active_cancer", "recently finished radiation", True, "yes", "leg_swelling", False),
    ("active_cancer", "being treated for a tumor", True, "yes", "leg_swelling", False),
    ("active_cancer", "no history of cancer", False, "no", "leg_swelling", True),
    ("active_cancer", "never diagnosed", False, "no", "leg_swelling", True),
    ("active_cancer", "no chemo or radiation", False, "no", "leg_swelling", True),
    ("active_cancer", "not currently", "unclear", "hedged", "leg_swelling", True),
    ("active_cancer", "in remission I think", "unclear", "hedged", "leg_swelling", True),
    ("active_cancer", "I hope not", "unclear", "hedged", "leg_swelling", True),

    # === paralysis_or_immobilization (Wells) ===
    ("paralysis_or_immobilization", "wearing a leg brace", True, "yes", "leg_swelling", False),
    ("paralysis_or_immobilization", "leg is in a splint", True, "yes", "leg_swelling", False),
    ("paralysis_or_immobilization", "can't move it well", True, "yes", "leg_swelling", False),
    ("paralysis_or_immobilization", "no cast or brace", False, "no", "leg_swelling", True),
    ("paralysis_or_immobilization", "leg moves fine", False, "no", "leg_swelling", False),
    ("paralysis_or_immobilization", "no immobilization", False, "no", "leg_swelling", True),
    ("paralysis_or_immobilization", "a bit stiff", "unclear", "hedged", "leg_swelling", True),
    ("paralysis_or_immobilization", "sometimes weak", "unclear", "hedged", "leg_swelling", True),
    ("paralysis_or_immobilization", "not paralyzed but sore", "unclear", "hedged", "leg_swelling", True),

    # === bedridden_or_surgery (Wells) ===
    ("bedridden_or_surgery", "had surgery 4 weeks ago", True, "yes", "leg_swelling", False),
    ("bedridden_or_surgery", "was in the hospital recently", True, "yes", "leg_swelling", False),
    ("bedridden_or_surgery", "been stuck in bed", True, "yes", "leg_swelling", False),
    ("bedridden_or_surgery", "no operations lately", False, "no", "leg_swelling", True),
    ("bedridden_or_surgery", "mobile and active", False, "no", "leg_swelling", False),
    ("bedridden_or_surgery", "no hospital stays", False, "no", "leg_swelling", True),
    ("bedridden_or_surgery", "does a short ER visit count", "unclear", "hedged", "leg_swelling", True),
    ("bedridden_or_surgery", "minor procedure maybe", "unclear", "hedged", "leg_swelling", True),
    ("bedridden_or_surgery", "unsure", "unclear", "hedged", "leg_swelling", True),

    # === localized_tenderness (Wells) ===
    ("localized_tenderness", "hurts if I press on it", True, "yes", "leg_swelling", False),
    ("localized_tenderness", "tender to touch", True, "yes", "leg_swelling", False),
    ("localized_tenderness", "sensitive when poked", True, "yes", "leg_swelling", False),
    ("localized_tenderness", "doesn't hurt when pressed", False, "no", "leg_swelling", True),
    ("localized_tenderness", "no tenderness", False, "no", "leg_swelling", True),
    ("localized_tenderness", "pain-free to touch", False, "no", "leg_swelling", False),
    ("localized_tenderness", "a little tender maybe", "unclear", "hedged", "leg_swelling", True),
    ("localized_tenderness", "only sometimes", "unclear", "hedged", "leg_swelling", True),
    ("localized_tenderness", "I think so", "unclear", "hedged", "leg_swelling", True),

    # === entire_leg_swollen (Wells) ===
    ("entire_leg_swollen", "swollen all over", True, "yes", "leg_swelling", False),
    ("entire_leg_swollen", "from thigh down to foot", True, "yes", "leg_swelling", False),
    ("entire_leg_swollen", "whole leg is puffy", True, "yes", "leg_swelling", False),
    ("entire_leg_swollen", "just the ankle", False, "no", "leg_swelling", False),
    ("entire_leg_swollen", "only my calf", False, "no", "leg_swelling", False),
    ("entire_leg_swollen", "localized to one spot", False, "no", "leg_swelling", False),
    ("entire_leg_swollen", "mostly calf but a little above", "unclear", "hedged", "leg_swelling", True),
    ("entire_leg_swollen", "not sure how high", "unclear", "hedged", "leg_swelling", True),
    ("entire_leg_swollen", "I think just part", "unclear", "hedged", "leg_swelling", True),

    # === calf_swelling_over_3cm (Wells) ===
    ("calf_swelling_over_3cm", "left calf is way bigger", True, "yes", "leg_swelling", False),
    ("calf_swelling_over_3cm", " noticeably larger than the other", True, "yes", "leg_swelling", False),
    ("calf_swelling_over_3cm", "swollen more than 3 cm", True, "yes", "leg_swelling", False),
    ("calf_swelling_over_3cm", "both calves same size", False, "no", "leg_swelling", False),
    ("calf_swelling_over_3cm", "not asymmetrical", False, "no", "leg_swelling", True),
    ("calf_swelling_over_3cm", "no extra swelling", False, "no", "leg_swelling", True),
    ("calf_swelling_over_3cm", "maybe slightly bigger", "unclear", "hedged", "leg_swelling", True),
    ("calf_swelling_over_3cm", "could be", "unclear", "hedged", "leg_swelling", True),
    ("calf_swelling_over_3cm", "hard to measure", "unclear", "hedged", "leg_swelling", True),

    # === pitting_edema (Wells) ===
    ("pitting_edema", "skin stays dimpled", True, "yes", "leg_swelling", False),
    ("pitting_edema", "dent remains after pressing", True, "yes", "leg_swelling", False),
    ("pitting_edema", "leaves an indent", True, "yes", "leg_swelling", False),
    ("pitting_edema", "no dent left behind", False, "no", "leg_swelling", True),
    ("pitting_edema", "pops right back", False, "no", "leg_swelling", False),
    ("pitting_edema", "elastic skin", False, "no", "leg_swelling", False),
    ("pitting_edema", "slight indent maybe", "unclear", "hedged", "leg_swelling", True),
    ("pitting_edema", "I'm not sure", "unclear", "hedged", "leg_swelling", True),
    ("pitting_edema", "a tiny pit", "unclear", "hedged", "leg_swelling", True),

    # === collateral_veins (Wells) ===
    ("collateral_veins", "new veins showing", True, "yes", "leg_swelling", False),
    ("collateral_veins", "spider veins appeared", True, "yes", "leg_swelling", False),
    ("collateral_veins", "veins more visible lately", True, "yes", "leg_swelling", False),
    ("collateral_veins", "no new veins", False, "no", "leg_swelling", True),
    ("collateral_veins", "veins look the same", False, "no", "leg_swelling", False),
    ("collateral_veins", "nothing new on the skin", False, "no", "leg_swelling", False),
    ("collateral_veins", "a couple maybe", "unclear", "hedged", "leg_swelling", True),
    ("collateral_veins", "I think so", "unclear", "hedged", "leg_swelling", True),
    ("collateral_veins", "not certain", "unclear", "hedged", "leg_swelling", True),

    # === chf_history (CHA₂DS₂-VASc) ===
    ("chf_history", "doctor said heart failure", True, "yes", "afib_stroke", False),
    ("chf_history", "congestive heart failure", True, "yes", "afib_stroke", False),
    ("chf_history", "my heart pumps poorly", True, "yes", "afib_stroke", False),
    ("chf_history", "heart is healthy", False, "no", "afib_stroke", False),
    ("chf_history", "no CHF history", False, "no", "afib_stroke", True),
    ("chf_history", "never diagnosed with heart failure", False, "no", "afib_stroke", True),
    ("chf_history", "not that I know of", "unclear", "hedged", "afib_stroke", True),
    ("chf_history", "I don't think so", "unclear", "hedged", "afib_stroke", True),
    ("chf_history", "maybe mild", "unclear", "hedged", "afib_stroke", True),

    # === hypertension (CHA₂DS₂-VASc) ===
    ("hypertension", "on blood pressure meds", True, "yes", "afib_stroke", False),
    ("hypertension", "high BP diagnosed", True, "yes", "afib_stroke", False),
    ("hypertension", "hypertensive", True, "yes", "afib_stroke", False),
    ("hypertension", "BP normal", False, "no", "afib_stroke", False),
    ("hypertension", "no high blood pressure", False, "no", "afib_stroke", True),
    ("hypertension", "not hypertensive", False, "no", "afib_stroke", True),
    ("hypertension", "borderline", "unclear", "hedged", "afib_stroke", True),
    ("hypertension", "sometimes elevated", "unclear", "hedged", "afib_stroke", True),
    ("hypertension", "not sure", "unclear", "hedged", "afib_stroke", True),

    # === stroke_tia_history (CHA₂DS₂-VASc) ===
    ("stroke_tia_history", "had a stroke previously", True, "yes", "afib_stroke", False),
    ("stroke_tia_history", "mini-stroke once", True, "yes", "afib_stroke", False),
    ("stroke_tia_history", "TIA in the past", True, "yes", "afib_stroke", False),
    ("stroke_tia_history", "no stroke history", False, "no", "afib_stroke", True),
    ("stroke_tia_history", "never had a TIA", False, "no", "afib_stroke", True),
    ("stroke_tia_history", "no brain clots", False, "no", "afib_stroke", True),
    ("stroke_tia_history", "maybe a small event", "unclear", "hedged", "afib_stroke", True),
    ("stroke_tia_history", "I think not", "unclear", "hedged", "afib_stroke", True),
    ("stroke_tia_history", "unsure", "unclear", "hedged", "afib_stroke", True),

    # === vascular_disease (CHA₂DS₂-VASc) ===
    ("vascular_disease", "had a stent placed", True, "yes", "afib_stroke", False),
    ("vascular_disease", "peripheral artery disease", True, "yes", "afib_stroke", False),
    ("vascular_disease", "coronary artery disease", True, "yes", "afib_stroke", False),
    ("vascular_disease", "no vascular problems", False, "no", "afib_stroke", True),
    ("vascular_disease", "arteries clean", False, "no", "afib_stroke", False),
    ("vascular_disease", "no stents or bypass", False, "no", "afib_stroke", True),
    ("vascular_disease", "family history only", "unclear", "hedged", "afib_stroke", True),  # family-history exclusion
    ("vascular_disease", "I don't think so", "unclear", "hedged", "afib_stroke", True),
    ("vascular_disease", "unsure", "unclear", "hedged", "afib_stroke", True),

    # === diabetes (CHA₂DS₂-VASc) ===
    ("diabetes", "type 2 diabetic", True, "yes", "afib_stroke", False),
    ("diabetes", "on metformin", True, "yes", "afib_stroke", False),
    ("diabetes", "high blood sugar", True, "yes", "afib_stroke", False),
    ("diabetes", "not diabetic", False, "no", "afib_stroke", True),
    ("diabetes", "no diabetes", False, "no", "afib_stroke", True),
    ("diabetes", "blood sugar normal", False, "no", "afib_stroke", False),
    ("diabetes", "prediabetic", "unclear", "hedged", "afib_stroke", True),
    ("diabetes", "borderline sugar", "unclear", "hedged", "afib_stroke", True),
    ("diabetes", "I think so", "unclear", "hedged", "afib_stroke", True),
]

# Verify count
assert len(PHRASES) == 153, f"Expected 153 phrases, got {len(PHRASES)}"
