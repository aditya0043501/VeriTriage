"""
Conversational extraction for leg swelling / Wells' DVT criteria.

Fully deterministic — no LLM calls. Uses explicit pattern tables in
rule_fallback.py plus quick-reply chip support.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple
from pydantic import BaseModel

from extraction.rule_fallback import (
    extract_age as _rf_extract_age,
    extract_wells_fields,
    WELLS_QUESTIONS,
    WELLS_REPHRASINGS,
    WELLS_PATTERNS,
    find_source_quote,
    detect_yes_no,
    detect_definition_request,
    build_definition_reply,
    age_in_range,
    has_out_of_range_age_number,
    INVALID_AGE_MESSAGE,
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

CRITERIA_KEYS = [
    "active_cancer", "paralysis_or_immobilization", "bedridden_or_surgery",
    "localized_tenderness", "entire_leg_swollen", "calf_swelling_over_3cm",
    "pitting_edema", "collateral_veins",
]


class LegSwellingData(BaseModel):
    """Structured data for Wells' DVT score (patient-knowable criteria)."""
    age: Optional[int] = None  # used for population scope check only
    active_cancer: Optional[bool] = None
    paralysis_or_immobilization: Optional[bool] = None
    bedridden_or_surgery: Optional[bool] = None
    localized_tenderness: Optional[bool] = None
    entire_leg_swollen: Optional[bool] = None
    calf_swelling_over_3cm: Optional[bool] = None
    pitting_edema: Optional[bool] = None
    collateral_veins: Optional[bool] = None
    retry_count: int = 0
    last_asked_field: Optional[str] = None
    # Per-field count of "not sure"-style answers, and fields we stopped
    # asking about after repeated unclear answers (see extraction_utils).
    unclear_counts: Dict[str, int] = {}
    unresolved_fields: List[str] = []
    # The patient's own words that triggered each positively-matched
    # criterion, for the explanation layer. Pure record-keeping — it never
    # affects extraction.
    source_quotes: Dict[str, str] = {}

    def is_complete(self) -> bool:
        return all(getattr(self, k) is not None for k in CRITERIA_KEYS)

    def get_missing_fields(self) -> List[str]:
        return [k for k in CRITERIA_KEYS if getattr(self, k) is None]


def get_opening_message() -> str:
    return "This tool evaluates leg swelling using a validated clinical scoring system. Your information is processed for this assessment only. Tell me about your leg in your own words."


def get_initial_question() -> str:
    return "Can you describe what you're noticing in your leg and when it started?"


def extract_and_update_data(
    conversation_history: List[Dict[str, str]],
    current_input: str,
    current_data: LegSwellingData
) -> Tuple[str, LegSwellingData, bool]:
    """Process patient input using deterministic extraction."""

    # Definition request ("what is pitting edema?") — answer it, then re-ask
    # the pending question. Does not consume an escalation attempt.
    term = detect_definition_request(current_input)
    if term:
        pending = WELLS_QUESTIONS.get(current_data.last_asked_field)
        if pending:
            return (build_definition_reply(term, pending), current_data, False)

    # Age regex fast-path (population scope only)
    if current_data.age is None:
        age_match = re.search(r'\b(\d{1,3})\s*(years? old|years?|yrs?|y/o)\b', current_input)
        if age_match:
            n = int(age_match.group(1))
            if age_in_range(n):
                current_data.age = n
        else:
            a = _rf_extract_age(current_input)
            if a is not None and age_in_range(a):
                current_data.age = a

    # Build full conversation context
    user_turns = [t["content"] for t in conversation_history if t["role"] == "user"]
    user_turns.append(current_input)
    combined = " ".join(user_turns)

    # Handle the opening question: a bare yes/ok or a descriptive answer means proceed
    if current_data.last_asked_field is None and not current_data.is_complete():
        from extraction.rule_fallback import detect_yes_no
        yn = detect_yes_no(current_input)
        if yn is True or _is_descriptive_input(current_input):
            current_data.last_asked_field = "age" if current_data.age is None else CRITERIA_KEYS[0]
            if current_data.last_asked_field == "age":
                return ("Got it. How old are you?", current_data, False)
            else:
                return (f"Got it. {WELLS_QUESTIONS[current_data.last_asked_field]}", current_data, False)

    # Apply bare yes/no to the last-asked field
    if apply_bare_yes_no(current_input, current_data, CRITERIA_KEYS):
        # A bare yes/no answer resolved the pending field — quote it verbatim
        f = current_data.last_asked_field
        if (f in CRITERIA_KEYS and getattr(current_data, f) is True
                and f not in current_data.source_quotes):
            current_data.source_quotes[f] = current_input.strip()

    # Run deterministic extraction
    missing = current_data.get_missing_fields()
    extracted, unclear = extract_wells_fields(combined, current_input, missing, current_data.last_asked_field)

    for k, v in extracted.items():
        if k in CRITERIA_KEYS and getattr(current_data, k) is None:
            setattr(current_data, k, v)
            if v is True and k not in current_data.source_quotes:
                current_data.source_quotes[k] = find_source_quote(
                    user_turns, current_input, k, WELLS_PATTERNS[k]
                )

    current_data.retry_count = 0

    if current_data.is_complete():
        current_data.last_asked_field = None
        return ("Thank you. I have what I need to assess your leg swelling.", current_data, True)

    still_missing = current_data.get_missing_fields()
    next_field = still_missing[0]

    previous_field = current_data.last_asked_field
    previous_answered = False
    if previous_field == "age":
        previous_answered = current_data.age is not None
    elif previous_field in CRITERIA_KEYS:
        previous_answered = getattr(current_data, previous_field) is not None

    current_data.last_asked_field = next_field

    base_question = WELLS_QUESTIONS[next_field]
    cue = ""
    if len(still_missing) <= 2:
        cue = " Just one or two more questions."
    if len(still_missing) <= 1:
        cue = " Just one more question."

    is_clarifying = not previous_answered and previous_field == next_field
    if is_clarifying:
        # Escalate rather than repeating the identical question: rephrase
        # once, then stop asking and move on.
        attempts = register_unclear_attempt(current_data, next_field)
        if attempts >= MAX_UNCLEAR_ATTEMPTS:
            mark_field_unresolved(current_data, next_field)
            if current_data.is_complete():
                current_data.last_asked_field = None
                return ("No problem — we'll leave that one as not established and note it for your doctor. "
                        "Thank you. I have what I need to assess your leg swelling.", current_data, True)
            next_field = current_data.get_missing_fields()[0]
            current_data.last_asked_field = next_field
            return ("No problem — we'll leave that one as not established and note it for your doctor. "
                    f"{WELLS_QUESTIONS[next_field]}", current_data, False)
        rephrased = WELLS_REPHRASINGS.get(next_field, base_question)
        response = f"No problem — let me put that a different way. {rephrased} (Yes / No / Not sure)"
    else:
        ack = ""
        if not extracted:
            ack = _acknowledge_input(current_input)
        response = f"{ack}{base_question}{cue}"

    if is_repeat_question(response, conversation_history, is_clarifying_reprompt=is_clarifying):
        logger.info(f"[leg_swelling_extractor] CIRCUIT BREAKER fired on stuck field='{next_field}'")
        advanced = force_advance(current_data, WELLS_QUESTIONS, CRITERIA_KEYS)
        if advanced:
            return (advanced[0], current_data, False)
        return ("Thank you. I have what I need to assess your leg swelling.", current_data, True)

    return (response, current_data, False)


def _is_descriptive_input(text: str) -> bool:
    """Return True if the input looks like a descriptive symptom statement
    or a simple confirmation to proceed."""
    t = text.lower().strip()
    if t in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
        return True
    descriptive_words = ["swollen", "tender", "pain", "hurt", "sore", "red", "warm", "calf", "leg", "vein", "dent", "surgery", "cast"]
    return any(w in t for w in descriptive_words) or len(t.split()) >= 6
