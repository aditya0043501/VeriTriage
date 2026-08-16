"""
Conversational extraction for CHA₂DS₂-VASc (atrial fibrillation stroke risk).

All criteria are patient-observable — no lab or imaging dependency.
The patient must already know they have atrial fibrillation; this extractor
confirms that context before proceeding.

Fully deterministic — no LLM calls. Uses explicit pattern tables in
rule_fallback.py plus quick-reply chip support.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple
from pydantic import BaseModel

from extraction.rule_fallback import (
    extract_age as _rf_extract_age,
    extract_chadsvasc_fields,
    CHADSVASC_QUESTIONS,
    detect_yes_no,
    _acknowledge_input,
)
from extraction.extraction_utils import (
    apply_bare_yes_no,
    is_repeat_question,
    force_advance,
)

logger = logging.getLogger("veritriage")

CRITERIA_KEYS = [
    "chf_history", "hypertension", "stroke_tia_history",
    "vascular_disease", "diabetes",
]


class AFibStrokeData(BaseModel):
    """Structured data for CHA₂DS₂-VASc score (patient-knowable criteria)."""
    afib_confirmed: Optional[bool] = None  # must be True before collecting other fields
    age: Optional[int] = None
    sex: Optional[str] = None  # "male" or "female"
    chf_history: Optional[bool] = None
    hypertension: Optional[bool] = None
    stroke_tia_history: Optional[bool] = None
    vascular_disease: Optional[bool] = None
    diabetes: Optional[bool] = None
    retry_count: int = 0
    last_asked_field: Optional[str] = None

    def is_complete(self) -> bool:
        return (
            self.afib_confirmed is True
            and self.age is not None
            and self.sex is not None
            and all(getattr(self, k) is not None for k in CRITERIA_KEYS)
        )

    def get_missing_fields(self) -> List[str]:
        if self.afib_confirmed is not True:
            return ["afib_confirmed"]
        missing = []
        if self.age is None:
            missing.append("age")
        if self.sex is None:
            missing.append("sex")
        for k in CRITERIA_KEYS:
            if getattr(self, k) is None:
                missing.append(k)
        return missing


def get_opening_message() -> str:
    return ("This tool assesses stroke risk for people with atrial fibrillation using the "
            "CHA₂DS₂-VASc scoring system, a validated clinical instrument. "
            "Your information is processed for this assessment only.")


def get_initial_question() -> str:
    return "Have you been diagnosed with atrial fibrillation by a doctor?"


def extract_and_update_data(
    conversation_history: List[Dict[str, str]],
    current_input: str,
    current_data: AFibStrokeData
) -> Tuple[str, AFibStrokeData, bool]:
    """Process patient input using deterministic extraction."""

    # Age regex fast-path
    if current_data.age is None:
        age_match = re.search(r'\b(\d{1,3})\s*(years? old|years?|yrs?|y/o)\b', current_input)
        if age_match:
            current_data.age = int(age_match.group(1))
        else:
            a = _rf_extract_age(current_input)
            if a is not None:
                current_data.age = a

    # Sex detection from input
    if current_data.sex is None:
        input_lower = current_input.lower()
        if any(w in input_lower for w in ["i'm a woman", "i'm female", "i am female", "female", "i'm a lady", "woman"]):
            current_data.sex = "female"
        elif any(w in input_lower for w in ["i'm a man", "i'm male", "i am male", "male", "i'm a guy", "man"]):
            current_data.sex = "male"

    # AFib confirmation gate — must be confirmed before any other collection
    if current_data.afib_confirmed is None:
        yn = detect_yes_no(current_input)
        if yn is True:
            current_data.afib_confirmed = True
            current_data.last_asked_field = "age"
            return ("Thank you for confirming. I'll ask a few questions to assess your stroke risk using the "
                    "CHA₂DS₂-VASc system. How old are you?", current_data, False)
        elif yn is False:
            current_data.afib_confirmed = False
            return ("This stroke risk assessment is designed for people who have been diagnosed with atrial "
                    "fibrillation. If you haven't been diagnosed but are experiencing irregular heartbeat, "
                    "palpitations, or related concerns, please consult a healthcare provider for evaluation. "
                    "You can also start a new assessment for leg swelling or sore throat if those are your concern.",
                    current_data, True)
        else:
            ack = _acknowledge_input(current_input)
            current_data.last_asked_field = "afib_confirmed"
            return (f"{ack}To use this assessment, I need to know: have you been diagnosed with atrial "
                    "fibrillation by a doctor? (Yes / No / Not sure)", current_data, False)

    if current_data.afib_confirmed is False:
        return ("This evaluation is for patients with confirmed atrial fibrillation. "
                "Start a new conversation for another assessment.", current_data, True)

    # Build full conversation context
    user_turns = [t["content"] for t in conversation_history if t["role"] == "user"]
    user_turns.append(current_input)
    combined = " ".join(user_turns)

    # Apply bare yes/no to the last-asked field
    apply_bare_yes_no(current_input, current_data, CRITERIA_KEYS)

    # Run deterministic extraction
    missing = current_data.get_missing_fields()
    extracted, unclear = extract_chadsvasc_fields(combined, current_input, missing, current_data.last_asked_field)

    for k, v in extracted.items():
        if k == "age":
            if current_data.age is None:
                current_data.age = v
        elif k == "sex":
            if current_data.sex is None:
                current_data.sex = v
        elif k in CRITERIA_KEYS:
            setattr(current_data, k, v)

    current_data.retry_count = 0

    if current_data.is_complete():
        current_data.last_asked_field = None
        return ("Thank you. I have what I need to assess your stroke risk.", current_data, True)

    still_missing = current_data.get_missing_fields()
    next_field = still_missing[0]

    previous_field = current_data.last_asked_field
    previous_answered = False
    if previous_field == "age":
        previous_answered = current_data.age is not None
    elif previous_field == "sex":
        previous_answered = current_data.sex is not None
    elif previous_field == "afib_confirmed":
        previous_answered = current_data.afib_confirmed is not None
    elif previous_field in CRITERIA_KEYS:
        previous_answered = getattr(current_data, previous_field) is not None

    current_data.last_asked_field = next_field

    base_question = CHADSVASC_QUESTIONS[next_field]
    cue = ""
    if len(still_missing) <= 1:
        cue = " Just one more question."

    if not previous_answered and previous_field == next_field:
        response = f"I'm not sure I understood — could you let me know: {base_question} (Yes / No / Not sure)"
    else:
        ack = ""
        if not extracted:
            ack = _acknowledge_input(current_input)
        response = f"{ack}{base_question}{cue}"

    is_clarifying = not previous_answered and previous_field == next_field
    if is_repeat_question(response, conversation_history, is_clarifying_reprompt=is_clarifying):
        logger.info(f"[afib_extractor] CIRCUIT BREAKER fired on stuck field='{next_field}'")
        advanced = force_advance(current_data, CHADSVASC_QUESTIONS, CRITERIA_KEYS, ["age", "sex"])
        if advanced:
            return (advanced[0], current_data, False)
        return ("Thank you. I have what I need to assess your stroke risk.", current_data, True)

    return (response, current_data, False)


def _is_descriptive_input(text: str) -> bool:
    """Return True if the input looks like a descriptive symptom/history statement."""
    t = text.lower().strip()
    descriptive_words = ["heart failure", "hypertension", "high blood pressure", "stroke", "tia", "diabetes", "vascular", "heart attack", "stent", "bypass", "age", "years old", "male", "female"]
    return any(w in t for w in descriptive_words) or len(t.split()) >= 6
