"""
Shared utilities for extraction modules.

These helpers address critical extraction-loop bugs:
- Bare yes/no attribution to the last-asked field
- Circuit breaker: never ask the same question twice in a row
- Category-switch detection for mid-conversation redirects
"""

import re
import logging
from typing import Optional, List, Dict, Tuple
from extraction.rule_fallback import detect_yes_no, _acknowledge_input, DEFINITION_PREFIX

logger = logging.getLogger("veritriage")


def apply_bare_yes_no(current_input: str, current_data, criteria_keys: List[str]) -> bool:
    """
    If the patient gives a bare yes/no and we know which field was last asked,
    apply it directly. Returns True if a field was updated.

    Only fires for SHORT answers (≤4 words) — complex sentences like
    "I haven't had surgery and I'm not bedridden" must go through the
    full negation-aware extraction path, not be shortcut by bare yes/no.
    """
    last_field = getattr(current_data, "last_asked_field", None)
    if not last_field or last_field not in criteria_keys:
        return False
    if getattr(current_data, last_field) is not None:
        # Already filled — don't overwrite
        return False

    # Only treat very short answers as bare yes/no
    if len(current_input.strip().split()) > 4:
        return False

    yn = detect_yes_no(current_input)
    if yn is None:
        return False

    setattr(current_data, last_field, yn)
    logger.info(f"[apply_bare_yes_no] Applied bare yes/no to last_asked_field={last_field}: {yn}")
    return True


# ---- "Not sure" clarification handling ----
#
# When a patient answers "not sure", re-asking the identical sentence is a
# dead end. Instead we escalate deterministically:
#   attempt 1 -> re-ask using a static, pre-written rephrasing for that field
#   attempt 2 -> stop asking, record the field as unresolved, and move on
# so the patient is never trapped in a repeat loop.

MAX_UNCLEAR_ATTEMPTS = 2


# Responses that deliberately restate a pending question. These must never be
# treated as an accidental repeat by either the extractor-level or the
# top-level circuit breaker — otherwise a legitimate clarification looks like
# a stuck loop and gets converted into an error.
CLARIFYING_RESPONSE_MARKERS = (
    "I'm not sure I understood",
    f"{DEFINITION_PREFIX}:",
    "No problem \u2014 let me put that a different way",
    "No problem \u2014 we'll leave that one as not established",
)


def is_clarifying_response(response: Optional[str]) -> bool:
    """True if ``response`` intentionally restates a pending question."""
    if not response:
        return False
    return response.lstrip().startswith(CLARIFYING_RESPONSE_MARKERS)


def register_unclear_attempt(current_data, field: str) -> int:
    """Increment and return the number of unclear answers seen for ``field``."""
    counts = dict(getattr(current_data, "unclear_counts", None) or {})
    counts[field] = counts.get(field, 0) + 1
    current_data.unclear_counts = counts
    return counts[field]


def mark_field_unresolved(current_data, field: str) -> None:
    """Give up on a field after repeated "not sure" answers.

    The field is set to False so scoring can proceed — False contributes no
    points, which is the conservative direction — and the field name is
    recorded so the report can state plainly that it was never established
    rather than implying the patient answered "no".
    """
    unresolved = list(getattr(current_data, "unresolved_fields", None) or [])
    if field not in unresolved:
        unresolved.append(field)
    current_data.unresolved_fields = unresolved
    if getattr(current_data, field, None) is None:
        setattr(current_data, field, False)
    logger.info(f"[mark_field_unresolved] field={field} marked unresolved after "
                f"{MAX_UNCLEAR_ATTEMPTS} unclear answers; scored as no-contribution")


def is_repeat_question(candidate: str, conversation_history: List[Dict[str, str]],
                       is_clarifying_reprompt: bool = False) -> bool:
    """
    Circuit breaker: check if the candidate question is (nearly) identical
    to the last bot message. If so, we must NOT ask it again.

    A clarifying re-prompt intentionally embeds the original question, so it is
    not treated as an accidental repeat when is_clarifying_reprompt=True.
    """
    if is_clarifying_reprompt:
        return False
    if not conversation_history:
        return False
    # Find the last assistant message
    last_bot = None
    for turn in reversed(conversation_history):
        if turn["role"] == "assistant":
            last_bot = turn["content"]
            break
    if not last_bot:
        return False

    # Normalize: lowercase, strip whitespace/punctuation
    def normalize(s):
        return re.sub(r'[^a-z0-9\s]', '', s.lower().strip())

    cand_norm = normalize(candidate)
    bot_norm = normalize(last_bot)

    if not cand_norm or not bot_norm:
        return False

    # Exact match after normalization
    if cand_norm == bot_norm:
        logger.info(f"[is_repeat_question] Exact repeat detected: '{last_bot[:120]}'")
        return True

    # One is a substring of the other (e.g., LLM added "Thank you. " prefix)
    if cand_norm in bot_norm or bot_norm in cand_norm:
        logger.info(f"[is_repeat_question] Substring repeat detected: candidate='{candidate[:120]}' last_bot='{last_bot[:120]}'")
        return True

    # High word overlap (>=80% of words shared)
    cand_words = set(cand_norm.split())
    bot_words = set(bot_norm.split())
    if not cand_words or not bot_words:
        return False
    overlap = len(cand_words & bot_words) / max(len(cand_words), len(bot_words))
    if overlap >= 0.8:
        logger.info(f"[is_repeat_question] High word overlap ({overlap:.2f}) detected: candidate='{candidate[:120]}' last_bot='{last_bot[:120]}'")
        return True

    return False


def force_advance(current_data, questions_map: Dict[str, str], criteria_keys: List[str],
                  extra_keys: List[str] = None) -> Optional[Tuple[str, str]]:
    """
    Force-advance to the next missing field's question.
    Returns (next_question, next_field) or None if all fields are filled.

    Used by the circuit breaker when the LLM tries to repeat a question.
    """
    all_keys = (extra_keys or []) + criteria_keys
    for k in all_keys:
        v = getattr(current_data, k, None)
        if v is None and k in questions_map:
            current_data.last_asked_field = k
            cue = ""
            missing_count = sum(1 for kk in all_keys if getattr(current_data, kk, None) is None)
            if missing_count <= 1:
                cue = " Just one more question."
            elif missing_count <= 2:
                cue = " Just one or two more questions."
            logger.info(f"[force_advance] Force-advanced from stuck field to next_field={k} (remaining_missing={missing_count})")
            return (questions_map[k] + cue, k)
    logger.info("[force_advance] No missing fields found; treating as complete")
    return None


# ---- Category-switch detection ----

_CATEGORY_SWITCH_PATTERNS = {
    "leg_swelling": [
        r"leg swelling", r"swollen leg", r"leg pain", r"assess my leg",
        r"check my leg", r"dvt", r"blood clot in (my )?leg",
        r"start (a )?leg", r"switch to leg",
    ],
    "sore_throat": [
        r"sore throat", r"throat pain", r"assess my throat",
        r"check my throat", r"strep", r"start (a )?sore throat",
        r"switch to sore throat", r"throat",
    ],
    "afib_stroke": [
        r"atrial fibrillation", r"afib", r"a-fib", r"stroke risk",
        r"assess my stroke", r"check my stroke", r"chadsvasc",
        r"start (an )?afib", r"switch to afib", r"blood thinner",
    ],
    "pneumonia": [
        r"pneumonia", r"chest infection", r"lung infection",
        r"assess my (cough|pneumonia)", r"check my (cough|pneumonia)",
        r"start (a )?pneumonia", r"switch to pneumonia", r"curb-?65",
    ],
}


def detect_category_switch(current_input: str, current_category: str) -> Optional[str]:
    """
    Detect if the patient is explicitly requesting a different assessment
    category mid-conversation.

    Returns the requested category name, or None if no switch is detected.
    """
    text = current_input.lower().strip()

    # Must contain an explicit switch intent phrase
    switch_intents = [
        "switch to", "start a new", "start an", "start a",
        "i want to assess", "i want to check", "i want to evaluate",
        "i'd like to assess", "i'd like to check", "i'd like to evaluate",
        "can i assess", "can i check", "can i evaluate",
        "i have atrial fibrillation", "i have a sore throat",
        "i have leg swelling", "my leg is swollen",
        "i have a swollen leg", "check my stroke risk",
        "assess my leg", "assess my throat", "assess my stroke",
    ]

    has_switch_intent = any(intent in text for intent in switch_intents)

    for cat, patterns in _CATEGORY_SWITCH_PATTERNS.items():
        if cat == current_category:
            continue
        for pat in patterns:
            if re.search(pat, text):
                # If there's a switch intent phrase OR the input is clearly
                # about a different body part (not just mentioning it in passing),
                # detect the switch
                if has_switch_intent:
                    logger.info(f"[detect_category_switch] switch_intent: from={current_category} to={cat} input='{current_input[:100]}'")
                    return cat
                # Also detect if the input is short and clearly about another category
                # (e.g., "I have atrial fibrillation and want to check my stroke risk")
                if len(text.split()) <= 15:
                    logger.info(f"[detect_category_switch] short clear switch: from={current_category} to={cat} input='{current_input[:100]}'")
                    return cat

    return None


def get_category_switch_message(requested_category: str, current_category: str) -> str:
    """Generate a confirmation message for a category switch request."""
    current_names = {
        "leg_swelling": "leg swelling",
        "sore_throat": "sore throat",
        "afib_stroke": "AFib stroke risk",
        "pneumonia": "pneumonia risk",
    }
    requested_names = {
        "leg_swelling": "the leg swelling (DVT risk) assessment",
        "sore_throat": "the sore throat (strep risk) assessment",
        "afib_stroke": "the AFib stroke risk assessment",
        "pneumonia": "the pneumonia risk (CURB-65) assessment",
    }
    return (f"It sounds like you'd like to switch to {requested_names.get(requested_category, requested_category)}. "
            f"Your {current_names.get(current_category, current_category)} answers so far won't be saved. "
            f"Would you like to start the {requested_names.get(requested_category, requested_category)} now? (yes or no)")
