"""
Population scope detection for VeriTriage.

The validated scores (Wells', Centor, CHA₂DS₂-VASc) have not been validated for
pregnant patients, minors, or significantly immunocompromised patients.
If extraction detects any of these, the conversation stops and redirects.
"""

import re
from typing import Optional

SCOPE_DISCLAIMER = (
    "This tool is designed for non-pregnant adults without significant "
    "immunocompromise. If you are pregnant, under 18, or immunocompromised, "
    "please consult a healthcare provider directly, as these scores have not "
    "been validated for your situation."
)

PREGNANCY_PATTERNS = [
    r"\bpregnan\w*\b",
    r"\bexpecting( a baby| a child)?\b",
    r"\bweeks along\b",
    r"\btrimester\b",
]

IMMUNOCOMPROMISE_PATTERNS = [
    r"\bimmunocompromis\w*\b",
    r"\bimmunosuppress\w*\b",
    r"\bchemo(therapy)?\b",
    r"\bhiv\b",
    r"\baids\b",
    r"\borgan transplant\b",
    r"\btransplant recipient\b",
]


def check_population_scope(message: str, age: Optional[int] = None) -> Optional[str]:
    """
    Check whether the patient falls outside the validated population.

    Args:
        message: The patient's latest free-text message
        age: The patient's age, if already extracted from a previous turn

    Returns:
        The redirect message if the patient is out of the validated population,
        None otherwise.
    """
    text = message.lower()

    # If age wasn't passed in, try to extract it from the message text
    # so that "I'm 14 years old" in the first message triggers the guard
    # immediately rather than on the next turn.
    if age is None:
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?\s*old|years?|yrs?|y/o)\b', text)
        if age_match:
            age = int(age_match.group(1))

    if age is not None and age < 18:
        return SCOPE_DISCLAIMER

    for pattern in PREGNANCY_PATTERNS:
        if re.search(pattern, text):
            return SCOPE_DISCLAIMER

    for pattern in IMMUNOCOMPROMISE_PATTERNS:
        if re.search(pattern, text):
            return SCOPE_DISCLAIMER

    return None
