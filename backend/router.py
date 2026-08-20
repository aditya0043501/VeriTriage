"""
Router — deterministic classification of patient complaints.

Classifies patient descriptions into:
  - leg_swelling    (DVT / Wells' criteria)
  - sore_throat     (Centor / McIsaac criteria)
  - afib_stroke     (CHA₂DS₂-VASc stroke risk in atrial fibrillation)
  - out_of_scope    (a specific complaint that is clearly not one of the 3)
  - vague           (not enough information to classify either way)

Vague input must NOT be treated as out_of_scope. The caller is expected to
ask a clarifying question and re-attempt classification with accumulated
context. Only after 2-3 failed clarification rounds should the system
declare scope limitations.
"""

import re
from typing import Literal

Category = Literal["leg_swelling", "sore_throat", "afib_stroke", "pneumonia", "out_of_scope", "vague"]


# ---- Vagueness detection (keyword fallback) ----

_VAGUE_PHRASES = [
    "not feeling okay", "not feeling well", "don't feel well", "feeling unwell",
    "feel weird", "feeling weird", "feel strange", "feeling strange",
    "something's off", "something is off", "something's wrong", "something is wrong",
    "don't feel right", "not feeling right", "i feel bad", "feeling bad",
    "i feel off", "feeling off", "not myself", "don't feel like myself",
    "under the weather", "just not right", "something doesn't feel right",
    "i'm scared", "i feel scared", "worried about", "anxious about",
    "i don't know", "not sure what's wrong", "can't explain",
]

# Dict order is match precedence (first hit wins). Pneumonia is checked
# first because substring matching is used: "phlegm" contains "leg", and
# without this order "lots of phlegm" would misroute to leg_swelling.
# No pneumonia keyword overlaps any other category's keyword list.
_SPECIFIC_KEYWORDS = {
    "pneumonia": ["pneumonia", "chest infection", "lung infection", "pneumococcal", "respiratory infection",
                  "cough", "fever", "short of breath", "phlegm", "green sputum", "sputum"],
    "leg_swelling": ["leg", "calf", "dvt", "clot", "thigh", "ankle", "shin", "swollen leg", "leg swollen", "leg pain", "leg swelling"],
    "sore_throat": ["throat", "swallow", "tonsil", "pharyng", "larynx", "sore throat"],
    "afib_stroke": ["afib", "atrial fibrillation", "atrial fib", "irregular heartbeat", "irregular heart", "palpitation", "cha2ds2", "chadsvasc", "stroke risk", "blood thinner", "anticoagulation", "a-fib", "a fib"],
}

_OUT_OF_SCOPE_INDICATORS = [
    "headache", "head ache", "migraine", "stomach", "abdomen", "abdominal",
    "nausea", "vomit", "diarrhea", "rash", "skin", "back pain", "backache",
    "ear", "eye", "vision", "tooth", "dental", "joint", "arthritis",
    "fever only", "cough only", "runny nose", "congestion", "sinus",
    "dizzy", "dizziness", "faint", "fainting", "seizure",
    "urine", "urinating", "bladder", "kidney",
    "hand", "arm", "shoulder", "neck pain", "wrist", "finger",
    "foot", "toe", "hip", "knee",
    "bleeding", "wound", "cut", "burn",
    "anxiety only", "depression", "mental", "sleep", "insomnia",
    "pregnancy", "pregnant",
    "chest pain", "chest", "heart attack", "heart pain", "cardiac",
]


def _is_vague(description: str) -> bool:
    text = description.lower().strip()
    if len(text.split()) <= 3 and not any(
        kw in text for kws in _SPECIFIC_KEYWORDS.values() for kw in kws
    ):
        if any(vp in text for vp in _VAGUE_PHRASES):
            return True
        if len(text.split()) <= 3:
            return True
    if any(vp in text for vp in _VAGUE_PHRASES):
        if not any(kw in text for kws in _SPECIFIC_KEYWORDS.values() for kw in kws):
            return True
    return False


def classify_complaint(patient_description: str) -> Category:
    """Deterministic complaint classification based on keyword rules."""
    text = patient_description.lower().strip()

    # Vague/greeting pre-check BEFORE any keyword routing: very short input
    # with no clinical content (e.g. "hi", "ok", "thanks") goes to the vague
    # clarifying flow and never reaches category matching. Inputs naming a
    # specific out-of-scope symptom (e.g. "headache") still route correctly.
    words = text.split()
    if len(words) <= 3:
        has_specific = any(k in text for kws in _SPECIFIC_KEYWORDS.values() for k in kws)
        has_out_of_scope = any(k in text for k in _OUT_OF_SCOPE_INDICATORS)
        if not has_specific and not has_out_of_scope:
            return "vague"  # type: ignore

    for category, keywords in _SPECIFIC_KEYWORDS.items():
        if any(k in text for k in keywords):
            return category  # type: ignore

    if any(k in text for k in _OUT_OF_SCOPE_INDICATORS):
        return "out_of_scope"

    if _is_vague(patient_description):
        return "vague"

    if len(text.split()) <= 5:
        return "vague"

    return "out_of_scope"


def get_out_of_scope_message() -> str:
    return (
        "This tool currently supports evaluation of leg swelling (blood clot risk), "
        "sore throat (strep risk), cough and fever (pneumonia risk), and stroke risk "
        "for atrial fibrillation patients. "
        "For other symptoms, please consult a healthcare provider directly."
    )


def get_vague_clarifying_question(round_num: int) -> str:
    if round_num == 1:
        return (
            "I'm sorry to hear that — can you tell me more about what you're "
            "feeling? Where in your body, and what does it feel like?"
        )
    elif round_num == 2:
        return (
            "Thank you. To help me understand, could you describe the main "
            "symptom that's bothering you most — is it related to your legs, "
            "your throat, or your heart, or somewhere else?"
        )
    else:
        return (
            "I want to make sure I'm understanding you correctly. Could you "
            "describe in a few words what specific symptom or sensation "
            "brought you here today?"
        )


def get_vague_escalation_message() -> str:
    return (
        "I want to help, but I'm having trouble understanding what symptoms "
        "you're experiencing. This tool currently supports evaluation of leg "
        "swelling, sore throat, cough and fever (pneumonia risk), and stroke risk "
        "for atrial fibrillation patients. "
        "If your symptoms relate to one of these, please describe them and we "
        "can begin. For other concerns, please consult a healthcare provider directly."
    )
