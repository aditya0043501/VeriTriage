"""
Conversational extraction for sore throat / Centor-McIsaac criteria.

Fully deterministic — no LLM calls. Uses explicit pattern tables in
rule_fallback.py plus quick-reply chip support.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple
from pydantic import BaseModel

from extraction.rule_fallback import (
    extract_age as _rf_extract_age,
    extract_centor_fields,
    CENTOR_QUESTIONS,
    CENTOR_REPHRASINGS,
    detect_yes_no,
    detect_definition_request,
    build_definition_reply,
    _acknowledge_input,
)
from extraction.extraction_utils import (
    apply_bare_yes_no,
    is_repeat_question,
    force_advance,
    register_unclear_attempt,
    mark_field_unresolved,
    MAX_UNCLEAR_ATTEMPTS,
)

logger = logging.getLogger("veritriage")

CRITERIA_KEYS = ["fever", "absence_of_cough", "tender_cervical_nodes", "tonsillar_exudate"]


class SoreThroatData(BaseModel):
    """Structured data for Centor/McIsaac score (patient-knowable criteria)."""
    age: Optional[int] = None  # required: McIsaac age modification
    fever: Optional[bool] = None
    absence_of_cough: Optional[bool] = None
    tender_cervical_nodes: Optional[bool] = None
    tonsillar_exudate: Optional[bool] = None
    retry_count: int = 0
    last_asked_field: Optional[str] = None
    # Per-field count of "not sure"-style answers, and fields we stopped
    # asking about after repeated unclear answers (see extraction_utils).
    unclear_counts: Dict[str, int] = {}
    unresolved_fields: List[str] = []

    def is_complete(self) -> bool:
        return self.age is not None and all(getattr(self, k) is not None for k in CRITERIA_KEYS)

    def get_missing_fields(self) -> List[str]:
        missing = [k for k in CRITERIA_KEYS if getattr(self, k) is None]
        if self.age is None:
            missing.append("age")
        return missing


def get_opening_message() -> str:
    return "This tool evaluates sore throat using a validated clinical scoring system. Your information is processed for this assessment only. Tell me about your sore throat in your own words."


def get_initial_question() -> str:
    return "Can you describe how your throat feels and when it started?"


def extract_and_update_data(
    conversation_history: List[Dict[str, str]],
    current_input: str,
    current_data: SoreThroatData
) -> Tuple[str, SoreThroatData, bool]:
    """Process patient input using deterministic extraction."""

    # Definition request ("what are tonsils?") — answer it, then re-ask the
    # pending question. Does not consume an escalation attempt.
    term = detect_definition_request(current_input)
    if term:
        pending = CENTOR_QUESTIONS.get(current_data.last_asked_field)
        if pending:
            return (build_definition_reply(term, pending), current_data, False)

    # Age regex fast-path (needed for McIsaac modification)
    if current_data.age is None:
        age_match = re.search(r'\b(\d{1,3})\s*(years? old|years?|yrs?|y/o)\b', current_input)
        if age_match:
            current_data.age = int(age_match.group(1))
        else:
            a = _rf_extract_age(current_input)
            if a is not None:
                current_data.age = a

    # Build full conversation context
    user_turns = [t["content"] for t in conversation_history if t["role"] == "user"]
    user_turns.append(current_input)
    combined = " ".join(user_turns)

    # Handle the opening question: a bare yes/ok or a descriptive answer means proceed
    if current_data.last_asked_field is None and not current_data.is_complete():
        yn = detect_yes_no(current_input)  # detect_yes_no is imported via rule_fallback
        if yn is True or _is_descriptive_input(current_input):
            # Acknowledge and move to first real question
            current_data.last_asked_field = "age" if current_data.age is None else CRITERIA_KEYS[0]
            if current_data.last_asked_field == "age":
                return ("Got it. How old are you?", current_data, False)
            else:
                return (f"Got it. {CENTOR_QUESTIONS[current_data.last_asked_field]}", current_data, False)

    # Apply bare yes/no to the last-asked field
    apply_bare_yes_no(current_input, current_data, CRITERIA_KEYS)

    # Run deterministic extraction
    missing = current_data.get_missing_fields()
    extracted, unclear = extract_centor_fields(combined, current_input, missing, current_data.last_asked_field)

    # Apply extracted values
    for k, v in extracted.items():
        if k == "age" and current_data.age is None:
            current_data.age = v
        elif k in CRITERIA_KEYS and getattr(current_data, k) is None:
            setattr(current_data, k, v)

    current_data.retry_count = 0

    if current_data.is_complete():
        current_data.last_asked_field = None
        return ("Thank you. I have what I need to assess your sore throat.", current_data, True)

    # Determine next question
    still_missing = current_data.get_missing_fields()
    next_field = still_missing[0]

    # Did the user just answer the previously-asked field?
    previous_field = current_data.last_asked_field
    previous_answered = False
    if previous_field == "age":
        previous_answered = current_data.age is not None
    elif previous_field in CRITERIA_KEYS:
        previous_answered = getattr(current_data, previous_field) is not None

    current_data.last_asked_field = next_field

    # Build response
    base_question = CENTOR_QUESTIONS[next_field]
    cue = ""
    if len(still_missing) <= 1:
        cue = " Just one more question."

    # Clarifying re-prompt only if the user did NOT answer the previous question
    # AND the next field is still unclear from their input.
    if not previous_answered and previous_field == next_field:
        # Escalate rather than repeating the identical question: rephrase
        # once, then stop asking and move on. `age` is required for the
        # McIsaac modifier and has no safe default, so it is never given up
        # on — it just keeps the clearer rephrasing.
        attempts = register_unclear_attempt(current_data, next_field)
        if attempts >= MAX_UNCLEAR_ATTEMPTS and next_field in CRITERIA_KEYS:
            mark_field_unresolved(current_data, next_field)
            if current_data.is_complete():
                current_data.last_asked_field = None
                return ("No problem — we'll leave that one as not established and note it for your doctor. "
                        "Thank you. I have what I need to assess your sore throat.", current_data, True)
            next_field = current_data.get_missing_fields()[0]
            current_data.last_asked_field = next_field
            return ("No problem — we'll leave that one as not established and note it for your doctor. "
                    f"{CENTOR_QUESTIONS[next_field]}", current_data, False)
        rephrased = CENTOR_REPHRASINGS.get(next_field, base_question)
        response = f"No problem — let me put that a different way. {rephrased} (Yes / No / Not sure)"
    else:
        ack = ""
        if not extracted:
            ack = _acknowledge_input(current_input)
        response = f"{ack}{base_question}{cue}"

    # Circuit breaker: never ask the same question twice in a row
    is_clarifying = not previous_answered and previous_field == next_field
    if is_repeat_question(response, conversation_history, is_clarifying_reprompt=is_clarifying):
        logger.info(f"[sore_throat_extractor] CIRCUIT BREAKER fired on stuck field='{next_field}'")
        advanced = force_advance(current_data, CENTOR_QUESTIONS, CRITERIA_KEYS, ["age"])
        if advanced:
            return (advanced[0], current_data, False)
        return ("Thank you. I have what I need to assess your sore throat.", current_data, True)

    return (response, current_data, False)


def _is_descriptive_input(text: str) -> bool:
    """Return True if the input looks like a descriptive symptom statement
    or a simple confirmation to proceed."""
    t = text.lower().strip()
    if t in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
        return True
    descriptive_words = ["hurt", "pain", "sore", "swollen", "red", "white", "patch", "fever", "cough", "throat", "neck", "gland", "tonsil"]
    return any(w in t for w in descriptive_words) or len(t.split()) >= 6
