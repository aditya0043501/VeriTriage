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
    CHADSVASC_REPHRASINGS,
    CHADSVASC_PATTERNS,
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
    "chf_history", "hypertension", "stroke_tia_history",
    "vascular_disease", "diabetes",
]

# The question the AFib gate asks before anything else is collected.
AFIB_GATE_QUESTION = "Have you been diagnosed with atrial fibrillation by a doctor?"

# Escalation attempt 2: a deliberately plainer, more direct phrasing, in case
# the clinical term itself is what's causing the uncertainty.
AFIB_DIRECT_QUESTION = (
    "Let me ask it as plainly as I can: has a doctor ever told you that you have an "
    "irregular heart rhythm called atrial fibrillation, or AFib? (Yes / No / Not sure)"
)

# Escalation attempt 3: stop. CHA₂DS₂-VASc is only meaningful for people with
# diagnosed atrial fibrillation, so guessing either way would produce a
# misleading score. End this assessment clearly rather than erroring out.
AFIB_CANNOT_PROCEED_MESSAGE = (
    "That's completely fine \u2014 it's not something everyone would know offhand. "
    "This particular assessment only produces a meaningful result for people who "
    "have been diagnosed with atrial fibrillation, so I shouldn't guess either way "
    "and won't calculate a score from an assumption. Your doctor or a copy of your "
    "medical records can confirm whether you have that diagnosis, and you're welcome "
    "to come back once you know. Nothing you've told me has been lost. In the "
    "meantime, you can start a new assessment for leg swelling or a sore throat if "
    "either of those is a concern."
)


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
    # Per-field count of "not sure"-style answers, and fields we stopped
    # asking about after repeated unclear answers (see extraction_utils).
    unclear_counts: Dict[str, int] = {}
    unresolved_fields: List[str] = []
    # The patient's own words that triggered each positively-matched
    # criterion, for the explanation layer (record-keeping only).
    source_quotes: Dict[str, str] = {}

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

    # Definition request ("what is atrial fibrillation?") — answer the
    # question the patient actually asked, then re-ask ours. Deliberately
    # before the AFib gate and before any hedge handling, and it neither
    # consumes an escalation attempt nor alters collected data.
    term = detect_definition_request(current_input)
    if term:
        pending = (AFIB_GATE_QUESTION if current_data.afib_confirmed is None
                   else CHADSVASC_QUESTIONS.get(current_data.last_asked_field))
        if pending:
            return (build_definition_reply(term, pending), current_data, False)

    # Age regex fast-path — bounds-checked: out-of-range numbers are never
    # stored. Unlike Centor (where age only drives the McIsaac modifier),
    # age is a SCORED criterion in CHA₂DS₂-VASc, so there is no give-up path
    # here — an unobtainable age keeps the rephrase, not a fabricated score.
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

    # Out-of-range age answer while age is pending -> dedicated message.
    # No give-up: CHA₂DS₂-VASc scores age directly, so proceeding without it
    # would fabricate points.
    if (current_data.age is None
            and current_data.last_asked_field == "age"
            and has_out_of_range_age_number(current_input)):
        return (INVALID_AGE_MESSAGE + " " + CHADSVASC_QUESTIONS["age"], current_data, False)

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
            # afib_confirmed has no safe default (unlike the scored criteria,
            # which can be treated as no-contribution), so it can never be
            # marked unresolved and skipped. But it must also never repeat
            # itself verbatim: doing so tripped the top-level circuit breaker,
            # which then surfaced "Something went wrong... let's restart" and
            # ended the session. Escalate deterministically instead:
            #   attempt 1 -> standard re-ask
            #   attempt 2 -> a more direct, plainer-worded question
            #   attempt 3 -> stop, explain why, and keep what we already have
            current_data.last_asked_field = "afib_confirmed"
            attempts = register_unclear_attempt(current_data, "afib_confirmed")
            if attempts == 1:
                ack = _acknowledge_input(current_input)
                return (f"{ack}To use this assessment, I need to know: have you been diagnosed with atrial "
                        "fibrillation by a doctor? (Yes / No / Not sure)", current_data, False)
            if attempts == 2:
                return (AFIB_DIRECT_QUESTION, current_data, False)
            return (AFIB_CANNOT_PROCEED_MESSAGE, current_data, True)

    if current_data.afib_confirmed is False:
        return ("This evaluation is for patients with confirmed atrial fibrillation. "
                "Start a new conversation for another assessment.", current_data, True)

    # Build full conversation context
    user_turns = [t["content"] for t in conversation_history if t["role"] == "user"]
    user_turns.append(current_input)
    combined = " ".join(user_turns)

    # Apply bare yes/no to the last-asked field
    if apply_bare_yes_no(current_input, current_data, CRITERIA_KEYS):
        # A bare yes/no answer resolved the pending field — quote it verbatim
        f = current_data.last_asked_field
        if (f in CRITERIA_KEYS and getattr(current_data, f) is True
                and f not in current_data.source_quotes):
            current_data.source_quotes[f] = current_input.strip()

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
            if v is True and k not in current_data.source_quotes:
                current_data.source_quotes[k] = find_source_quote(
                    user_turns, current_input, k, CHADSVASC_PATTERNS[k]
                )

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
        # Escalate rather than repeating the identical question: rephrase
        # once, then stop asking and move on. age/sex/afib_confirmed are
        # required for scoring and have no safe default, so they are never
        # given up on — they just keep the clearer rephrasing.
        attempts = register_unclear_attempt(current_data, next_field)
        if attempts >= MAX_UNCLEAR_ATTEMPTS and next_field in CRITERIA_KEYS:
            mark_field_unresolved(current_data, next_field)
            if current_data.is_complete():
                current_data.last_asked_field = None
                return ("No problem — we'll leave that one as not established and note it for your doctor. "
                        "Thank you. I have what I need to assess your stroke risk.", current_data, True)
            next_field = current_data.get_missing_fields()[0]
            current_data.last_asked_field = next_field
            return ("No problem — we'll leave that one as not established and note it for your doctor. "
                    f"{CHADSVASC_QUESTIONS[next_field]}", current_data, False)
        rephrased = CHADSVASC_REPHRASINGS.get(next_field, base_question)
        response = f"No problem — let me put that a different way. {rephrased} (Yes / No / Not sure)"
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
