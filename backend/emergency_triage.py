"""
Emergency triage gate — deterministic keyword check, front door only.

Called BEFORE any routing, extraction, scoring, or state mutation. If any
trigger matches, the caller returns the fixed emergency message immediately
and halts all normal processing.

No ML, no regex complexity, no conversation state. Trigger list and message
text are fixed and human-approved; do not extend without review.
"""

import logging
from typing import Optional

logger = logging.getLogger("veritriage")

EMERGENCY_MESSAGE = (
    "\U0001f6a8 This sounds like it could be a medical emergency. Please call your "
    "local emergency number (e.g., 911, 999, 112) or go to the nearest emergency "
    "department immediately. This tool is not designed for emergency situations."
)

# Multi-word / combined triggers are checked first (group 1 is a two-part
# combination: chest pain + breathing difficulty), then single-keyword groups.
# Case-insensitive containment, first match wins.

_CHEST_PAIN_BREATHING = (
    "can't breathe", "cant breathe", "short of breath", "breathing", "breathless",
)

_TRIGGER_GROUPS = [
    # 2. Cardiac
    ("heart attack", "cardiac arrest"),
    # 3. Stroke — acuity-framed phrases only ("happening now"), so history
    # ("I had a stroke two years ago") and pathway entry ("stroke risk")
    # do not halt the AFib assessment.
    ("having a stroke", "think it's a stroke", "think i'm having a stroke",
     "signs of a stroke", "facial droop", "arm weakness", "slurred speech",
     "can't speak"),
    # 4. Breathing
    ("can't breathe", "cant breathe", "suffocating", "choking"),
    # 5. Bleeding
    ("severe bleeding", "bleeding out"),
    # 7. Seizure
    ("seizure", "convulsing", "shaking uncontrollably"),
    # 8. Anaphylaxis
    ("allergic reaction", "anaphylaxis", "throat closing"),
    # 9. Suicidal ideation
    ("suicide", "kill myself", "want to die", "suicidal"),
]

_UNCONSCIOUS_MARKERS = ("unconscious", "passed out")
_FAINTED_NOT_WAKING = ("won't wake", "wont wake", "not waking")


def get_emergency_message(user_text: str) -> Optional[str]:
    """Return the fixed emergency message if any trigger matches, else None."""
    t = user_text.lower()

    # 1. "chest pain" AND breathing difficulty (combined trigger, checked first)
    if "chest pain" in t and any(m in t for m in _CHEST_PAIN_BREATHING):
        logger.warning(f"[emergency_triage] TRIGGERED: chest pain + breathing — '{user_text[:80]}'")
        return EMERGENCY_MESSAGE

    # Groups 2-5, 7-9: any single keyword in the group
    for group in _TRIGGER_GROUPS:
        for keyword in group:
            if keyword in t:
                logger.warning(f"[emergency_triage] TRIGGERED: '{keyword}' — '{user_text[:80]}'")
                return EMERGENCY_MESSAGE

    # 6. "unconscious" / "passed out", or "fainted" AND not waking
    if any(m in t for m in _UNCONSCIOUS_MARKERS):
        logger.warning(f"[emergency_triage] TRIGGERED: unconscious/passed out — '{user_text[:80]}'")
        return EMERGENCY_MESSAGE
    if "fainted" in t and any(m in t for m in _FAINTED_NOT_WAKING):
        logger.warning(f"[emergency_triage] TRIGGERED: fainted + not waking — '{user_text[:80]}'")
        return EMERGENCY_MESSAGE

    return None
