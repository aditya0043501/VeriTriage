"""
Deterministic rule-based extraction.

This module contains the ONLY extraction logic used by VeriTriage. There are
no LLM calls here. The philosophy is:

- Extract only what can be matched with explicit, testable rules.
- Never guess a clinical score.
- If an answer is unclear, return an explicit "unclear" signal so the caller
can ask for clarification (with quick-reply chips).
"""

import re
from typing import Dict, List, Optional, Tuple, Union


# ---- Shared utilities ----

def _has_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(p in t for p in patterns)


def _has_none(text: str, patterns: List[str]) -> bool:
    return not _has_any(text, patterns)


# ---- Negation handling ----
#
# A shared helper to prevent negated descriptions (e.g. "temperature was
# normal", "never had a TIA") from being scored as affirmative just because
# a positive keyword appears as a substring.

NEGATORS = {
    "not", "never", "no", "normal", "without", "isn't", "aren't", "wasn't",
    "weren't", "don't", "doesn't", "didn't", "haven't", "hasn't", "hadn't",
    "cannot", "can't", "won't", "wouldn't", "shouldn't", "couldn't", "none",
    "neither", "nor", "free", "absent",
}

# ---- Field polarity ----
#
# Most fields are normal-polarity: "no <keyword>" means the criterion is
# absent (False). A few fields are inverted-polarity: "no <keyword>" means
# the criterion is PRESENT (True). The negation scan must be aware of this
# so it flips in the correct direction.
#
# Currently the only inverted field is absence_of_cough (Centor), where
# "no cough" means the criterion (absence of cough) is satisfied = True.
#
# When a negated positive keyword is found:
#   - Normal field  -> False (criterion absent)
#   - Inverted field -> True  (criterion present)

INVERTED_FIELDS = frozenset({"absence_of_cough"})

# ---- "Ruled out" family (NegEx-derived phrase-aware negation) ----
#
# Double-negation forms are checked FIRST: they are more specific and contain
# the completed forms as substrings ("not ruled out" ⊃ "ruled out"), so the
# more-specific reading must win (see docs/negation-pattern-diff-proposal.md §3.1).

# "not ruled out" = condition remains possible -> unclear, never a confident False.
RULED_OUT_POSSIBLE = [
    "not ruled out", "not been ruled out", "did not rule out",
]

# Completed negation: the condition was excluded -> False (True for inverted fields).
RULED_OUT_NEGATED = [
    "was ruled out", "is ruled out", "has been ruled out",
    "have been ruled out", "rules out", "ruled out",
]

# ---- Scope termination (NegEx [CONJ] family, minimal subset) ----
#
# A negation trigger's scope does not cross a conjunction like "but" into a
# later clause. E.g. "I have no cough, but I do have a fever" — the "no"
# belongs to "cough" and must not suppress the affirmative "fever" clause
# that follows "but". This is scoped ONLY to "but" for now (see
# docs/negation-pattern-diff-proposal.md §3.1 / Stage 2c).
SCOPE_TERMINATORS = ["but"]


def _text_after_terminator(text: str, after_pos: int) -> Optional[str]:
    """Return the text following the first scope terminator that appears at
    or after ``after_pos`` in ``text``, or None if no terminator is found."""
    earliest = None
    for term in SCOPE_TERMINATORS:
        pos = text.find(term, after_pos)
        if pos != -1 and (earliest is None or pos < earliest[0]):
            earliest = (pos, pos + len(term))
    if earliest is None:
        return None
    return text[earliest[1]:]


def _no_pattern_scope_ends_before_yes(text: str, neg_pattern: str, field: str) -> bool:
    """Return True if a scope terminator separates ``neg_pattern`` from a
    later affirmative (YES_PATTERNS) match for ``field`` — meaning the
    negation's scope ends before that later clause, so it must not be
    treated as a negative answer for this field."""
    m = re.search(rf"\b{re.escape(neg_pattern)}\b", text)
    if not m:
        return False
    after_term = _text_after_terminator(text, m.end())
    if after_term is None:
        return False
    for yp in YES_PATTERNS.get(field, []):
        if re.search(rf"\b{re.escape(yp)}\b", after_term):
            return True
    return False


def _is_negated_context(text: str, keyword: str, window: int = 4) -> bool:
    """Return True if a negator appears within ``window`` words before or
    after the first occurrence of ``keyword`` in ``text``.

    This is the shared negation-scan step used across all fields to catch
    false-positive keyword matches like "temperature was normal" (fever) or
    "never had a TIA" (stroke history).

    Scope termination: a negator BEFORE the keyword does not count if a
    scope terminator (e.g. "but") falls between the negator and the
    keyword — the negator's scope ends at the terminator and does not
    cross into the following clause.
    """
    t = text.lower()
    kw = keyword.lower()
    idx = t.find(kw)
    if idx == -1:
        return False
    before_words = t[:idx].split()
    after_words = t[idx + len(kw):].split()
    before_window = before_words[-window:]
    # If a scope terminator appears in the before-window, only negators
    # after the terminator (i.e. in the same clause as the keyword) count.
    term_cut = 0
    for i, tok in enumerate(before_window):
        clean = re.sub(r"[^a-z']", "", tok)
        if clean in SCOPE_TERMINATORS:
            term_cut = i + 1
    nearby = before_window[term_cut:] + after_words[:window]
    for tok in nearby:
        clean = re.sub(r"[^a-z']", "", tok)
        if clean in NEGATORS or clean.endswith("n't"):
            return True
    return False


def _positive_keyword_hit(text: str, patterns: List[str]) -> bool:
    """Return True if any pattern matches in ``text`` AND is not negated.

    This replaces the old ``cur_hit and not cur_is_no`` idiom, which missed
    negated descriptions because ``detect_yes_no`` returns ``None`` (not
    ``False``) for phrases like "BP normal" — there's no explicit "no" word,
    just a negated positive keyword.
    """
    t = text.lower()
    for p in patterns:
        if p in t and not _is_negated_context(t, p):
            return True
    return False


# ---- Hedged-input detection ----

UNCERTAINTY_MARKERS = [
    "not sure", "don't know", "dont know", "don't really know",
    "dont really know", "not certain", "unclear", "unsure", "maybe",
    "i guess", "i think", "i suppose", "i don't think", "possibly",
    "might be", "could be", "hard to say", "can't tell", "cant tell",
    "no idea", "i doubt", "a little", "slight", "slightly", "sorta",
    "kinda", "kind of", "sort of", "somewhat", "a tiny", "a bit",
]


def _is_hedged(text: str) -> bool:
    """Return True if the text contains uncertainty/hedging markers.

    Hedged input must never resolve to a confident True — it should fall
    through to ``unclear`` so the caller can re-prompt with chips.
    """
    t = text.lower().strip()
    if detect_yes_no(t) is None:
        for m in UNCERTAINTY_MARKERS:
            if m in t:
                return True
    return False


def find_source_quote(
    user_turns: List[str],
    current_input: str,
    field: str,
    patterns: List[str],
) -> str:
    """Find the patient's own words that triggered a positive criterion match.

    Pure sidecar lookup: it runs AFTER extraction has already decided the
    field is True, and never influences that decision. Priority:
      1. An explicit yes-style answer to the pending question (current input)
      2. A keyword hit in the current message
      3. The earliest earlier user turn containing the match
      4. Fallback: the current input verbatim
    """
    if _resolve_yes_no(field, current_input) is True:
        return current_input.strip()
    if _positive_keyword_hit(current_input, patterns):
        return current_input.strip()
    for turn in user_turns:
        if turn is not current_input and _positive_keyword_hit(turn, patterns):
            return turn.strip()
    return current_input.strip()


# ---- Definition requests ("what does X mean?") ----
#
# "I don't know what atrial fibrillation is" is NOT the same as "I don't know
# if I have atrial fibrillation". The first is a request for a definition; the
# second is genuine uncertainty about oneself. Both contain "don't know", so
# UNCERTAINTY_MARKERS catches them equally and the patient asking what a word
# means just gets the same question repeated back at them.
#
# These patterns require "what"-style framing, so "I don't know IF I have X"
# still falls through to the hedge path as before.

DEFINITION_PREFIX = "Quick definition"

_DEFINITION_REQUEST_PATTERNS = [
    r"\bwhat(?:'s|s| is| are| does| do)\b",
    r"\bwhat do you mean\b",
    r"\b(?:don'?t|dont|do not) know what\b",
    r"\b(?:never|not) heard of\b",
    r"\bno idea what\b",
    r"\bdefine\b",
    r"\bmeaning of\b",
    r"\bmean(?:s|ing)?\?",
    r"\bexplain\b",
    r"\bconfused (?:about|by)\b",
]

# Plain-language definitions for the medical terms used in WELLS_QUESTIONS,
# CENTOR_QUESTIONS, CHADSVASC_QUESTIONS and their rephrasings/labels.
# Static and hand-written, like every other patient-facing string here.
# Each is descriptive only — it defines a term, it does not tell the patient
# whether they have the thing.
TERM_DEFINITIONS = {
    "atrial_fibrillation": (
        "Atrial fibrillation — often shortened to AFib — is a heart rhythm that is "
        "irregular, and often faster than normal. People sometimes notice it as a "
        "fluttering or racing heartbeat, but many people feel nothing at all. It is "
        "diagnosed by a doctor using a heart tracing called an ECG or EKG."
    ),
    "tia": (
        "A TIA, or transient ischaemic attack, is often called a mini-stroke. It is "
        "when the blood supply to part of the brain is briefly interrupted, causing "
        "stroke-like symptoms — such as weakness, numbness, or trouble speaking — "
        "that go away again, usually within minutes to a day."
    ),
    "stroke": (
        "A stroke happens when the blood supply to part of the brain is cut off, "
        "which can cause lasting effects such as weakness or numbness on one side, "
        "or difficulty speaking. It is diagnosed and treated in hospital."
    ),
    "heart_failure": (
        "Heart failure — sometimes called congestive heart failure or CHF — means the "
        "heart isn't pumping as strongly as it should. It often causes breathlessness, "
        "tiredness, and swelling in the legs or ankles. It is diagnosed by a doctor."
    ),
    "hypertension": (
        "Hypertension is the medical word for high blood pressure — blood pushing "
        "against the walls of your arteries harder than it should. It usually causes "
        "no symptoms, so it is normally found by having your blood pressure measured."
    ),
    "vascular_disease": (
        "Vascular disease means a problem with the arteries that carry blood around "
        "your body — for example a previous heart attack, a stent or bypass operation, "
        "or narrowed arteries in the legs."
    ),
    "diabetes": (
        "Diabetes is a condition where the level of sugar (glucose) in the blood is "
        "too high. It is usually managed with diet, tablets such as metformin, or "
        "insulin, and is diagnosed by a doctor with a blood test."
    ),
    "dvt": (
        "A DVT, or deep vein thrombosis, is a blood clot that forms in one of the deep "
        "veins, usually in the leg. It can cause swelling, pain, and warmth in that leg."
    ),
    "pitting_edema": (
        "Pitting edema is swelling that holds the shape of your finger for a moment "
        "after you press on it — you press, lift your finger, and a small dent stays "
        "behind before slowly filling back in."
    ),
    "collateral_veins": (
        "Collateral veins are small veins near the surface of the skin that become more "
        "visible when a deeper vein is blocked and blood is rerouted through them. They "
        "look like new veins that weren't there before, and are not the same as varicose "
        "veins."
    ),
    "paralysis": (
        "Paralysis means being unable to move a part of the body. A milder version, "
        "sometimes called paresis, means the limb is weak but can still move a little."
    ),
    "immobilization": (
        "Immobilization means a limb has been kept still — for example in a plaster "
        "cast, a splint, or a rigid brace — so it couldn't move normally."
    ),
    "bedridden": (
        "Bedridden means having to stay in bed more or less continuously, rather than "
        "getting up and moving around as usual."
    ),
    "tonsils": (
        "Your tonsils are the two soft lumps of tissue at the back of your throat, one "
        "on each side. You can usually see them with a mirror and a light if you open "
        "your mouth wide."
    ),
    "exudate": (
        "Exudate is the white or yellowish coating or patches that can appear on the "
        "tonsils during some throat infections."
    ),
    "lymph_nodes": (
        "Lymph nodes — sometimes called glands — are small lumps that are part of your "
        "immune system. The ones in the front of your neck, under the jaw, can become "
        "swollen and tender during a throat infection."
    ),
    "strep_throat": (
        "Strep throat is a sore throat caused by a bacterium called group A "
        "streptococcus, rather than by a virus. It is confirmed with a throat swab."
    ),
    "d_dimer": (
        "A D-dimer is a blood test that looks for fragments left behind when a blood "
        "clot breaks down. It is ordered and interpreted by a clinician."
    ),
}

# Term fragments mapped to definition keys. Ordered longest-first at match
# time so "atrial fibrillation" wins over "fibrillation", and common
# misspellings of the long terms are covered by matching a distinctive stem
# ("fibrillation" catches "artrial fibrillation").
_TERM_FRAGMENTS = {
    "atrial fibrillation": "atrial_fibrillation",
    "fibrillation": "atrial_fibrillation",
    "atrial fib": "atrial_fibrillation",
    "afib": "atrial_fibrillation",
    "a-fib": "atrial_fibrillation",
    "a fib": "atrial_fibrillation",
    "transient ischemic": "tia",
    "transient ischaemic": "tia",
    "mini-stroke": "tia",
    "mini stroke": "tia",
    "tia": "tia",
    "stroke": "stroke",
    "congestive heart failure": "heart_failure",
    "heart failure": "heart_failure",
    "chf": "heart_failure",
    "hypertension": "hypertension",
    "high blood pressure": "hypertension",
    "peripheral artery disease": "vascular_disease",
    "vascular disease": "vascular_disease",
    "vascular": "vascular_disease",
    "diabetes": "diabetes",
    "diabetic": "diabetes",
    "deep vein thrombosis": "dvt",
    "dvt": "dvt",
    "pitting edema": "pitting_edema",
    "pitting oedema": "pitting_edema",
    "pitting": "pitting_edema",
    "edema": "pitting_edema",
    "oedema": "pitting_edema",
    "collateral vein": "collateral_veins",
    "collateral": "collateral_veins",
    "paralysis": "paralysis",
    "paresis": "paralysis",
    "immobilization": "immobilization",
    "immobilisation": "immobilization",
    "splint": "immobilization",
    "bedridden": "bedridden",
    "bed ridden": "bedridden",
    "tonsillar": "tonsils",
    "tonsils": "tonsils",
    "tonsil": "tonsils",
    "exudate": "exudate",
    "lymph nodes": "lymph_nodes",
    "lymph node": "lymph_nodes",
    "cervical nodes": "lymph_nodes",
    "adenopathy": "lymph_nodes",
    "strep throat": "strep_throat",
    "streptococcal": "strep_throat",
    "strep": "strep_throat",
    "d-dimer": "d_dimer",
    "d dimer": "d_dimer",
}


def detect_definition_request(text: str) -> Optional[str]:
    """Return a definition-key if ``text`` is asking what a medical term means.

    Returns None for genuine self-uncertainty ("I don't know if I have X"),
    which must keep flowing to the existing hedge path.
    """
    t = text.lower().strip()
    if not any(re.search(p, t) for p in _DEFINITION_REQUEST_PATTERNS):
        return None
    # Longest fragment first so specific terms beat their own substrings.
    for fragment in sorted(_TERM_FRAGMENTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(fragment)}\b", t):
            return _TERM_FRAGMENTS[fragment]
    return None


def build_definition_reply(term_key: str, question: str) -> str:
    """Answer the definition request, then re-ask the pending question."""
    definition = TERM_DEFINITIONS[term_key]
    return f"{DEFINITION_PREFIX}: {definition}\n\nWith that in mind — {question}"


# ---- Family-history exclusion ----
#
# CHA₂DS₂-VASc (and most clinical scores) score PERSONAL history, not family
# history. "My father had a heart attack" does NOT mean the patient has
# vascular disease. This helper detects family-history framing so the
# extractor can exclude these from personal-history YES matches.

FAMILY_HISTORY_MARKERS = [
    "family history", "runs in the family", "runs in my family",
    "my mother had", "my father had", "my mom had", "my dad had",
    "my parents had", "my parent had", "my brother had", "my sister had",
    "my sibling had", "grandparent", "grandmother", "grandfather",
    "grandma had", "grandpa had", "aunt had", "uncle had",
    "family member", "relatives", "my family",
    "hereditary", "inherited", "genetic",
]


def _is_family_history(text: str) -> bool:
    """Return True if the text frames a condition as family history rather
    than personal history."""
    t = text.lower()
    return any(m in t for m in FAMILY_HISTORY_MARKERS)


def _acknowledge_input(text: str) -> str:
    """Brief acknowledgment for vague but valid input. Keeps flow open."""
    t = text.lower().strip()
    if any(w in t for w in ["since", "yesterday", "today", "days", "hours", "week", "ago", "started"]):
        return "Got it — that timing helps. "
    if any(w in t for w in ["hurts", "painful", "pain", "ache", "sore", "uncomfortable", "bother"]):
        return "I understand that's uncomfortable. "
    if any(w in t for w in ["don't know", "not sure", "maybe", "unsure", "unclear"]):
        return "That's okay — we can work with that. "
    if any(w in t for w in ["yes", "yeah", "correct", "right"]):
        return "Thank you. "
    if any(w in t for w in ["no", "nope"]):
        return "Understood. "
    return ""


def extract_age(text: str) -> Optional[int]:
    """Extract an age number from free text. Returns None if not found."""
    m = re.search(r'\b(\d{1,3})\s*(?:years? old|years?|yrs?|y/o)\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "I'm 55" or "I am 45" style
    m = re.search(r"\bi\s*(?:'m|am)\s+(\d{1,3})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def detect_yes_no(text: str) -> Optional[bool]:
    """Detect an explicit yes/no stance in text. Returns None if ambiguous.

    Uncertainty expressions (e.g. "I don't really know", "not sure", "maybe")
    are checked FIRST and always return None, so vague answers don't get
    misinterpreted as a "no".
    """
    t = text.lower().strip()

    # Uncertainty markers — check first, before any yes/no detection
    uncertainty_markers = [
        "not sure", "don't know", "dont know", "don't really know",
        "dont really know", "not certain", "unclear", "unsure", "maybe",
        "i guess", "i think", "i suppose", "i don't think", "possibly",
        "might be", "could be", "hard to say", "can't tell", "cant tell",
        "no idea", "i doubt",
    ]
    for um in uncertainty_markers:
        if um in t:
            return None

    yes_markers = ["yes", "yeah", "yep", "yup", "sure", "i do", "that's right", "correct", "affirm", "absolutely"]
    no_markers = ["no", "nope", "not really", "i don't", "i do not", "don't have", "negative", "never"]
    # Strong negation first (require word boundary after marker).
    # Use \b so punctuation (comma, period) doesn't block matching:
    # "yes, I have" should match, "no." at end of sentence should match.
    for nm in no_markers:
        if re.search(rf"\b{re.escape(nm)}\b", t):
            return False
    for ym in yes_markers:
        if re.search(rf"\b{re.escape(ym)}\b", t):
            return True
    return None


# ---- Explicit yes/no pattern tables per field ----
#
# Each field has a table of positive and negative pattern lists.
# The extraction functions use these to decide True, False, or "unclear".

YES_PATTERNS = {
    "fever": ["yes", "yeah", "yep", "yup", "sure", "i do", "i have", "i am", "i'm ", "that's right", "correct", "absolutely", "i believe so", "i suppose so"],
    "absence_of_cough": ["no cough", "don't have a cough", "do not have a cough", "without cough", "not coughing", "i'm not coughing", "no", "cough free", "cough-free", "haven't been coughing", "haven't coughed", "haven't had a cough", "no coughing", "without a cough"],
    "tender_cervical_nodes": ["yes", "yeah", "yep", "yup", "i have", "they are", "they're tender", "they're swollen", "they hurt"],
    "tonsillar_exudate": ["yes", "yeah", "yep", "yup", "i see", "i have", "there are", "white patches", "white spots", "looks pus-like", "my tonsils have coating"],
    "active_cancer": ["yes", "yeah", "yep", "yup", "i have", "i am", "i'm being treated", "currently", "within 6 months", "malignancy"],
    "paralysis_or_immobilization": ["yes", "yeah", "yep", "yup", "i have", "i am", "i can't move", "cast", "splint", "in a brace", "can't feel my leg"],
    "bedridden_or_surgery": ["yes", "yeah", "yep", "yup", "i have", "i was", "bedridden", "surgery", "operation", "i've been in bed"],
    "localized_tenderness": ["yes", "yeah", "yep", "yup", "it is", "it's tender", "hurts to touch", "painful", "hurts when touched", "very sore"],
    "entire_leg_swollen": ["yes", "yeah", "yep", "yup", "whole leg", "entire leg", "all of it", "full leg", "all the way up", "from hip to ankle", "whole limb"],
    "calf_swelling_over_3cm": ["yes", "yeah", "yep", "yup", "it is", "it's bigger", "noticeably", "more than 3", "clearly larger", "noticeable difference"],
    "pitting_edema": ["yes", "yeah", "yep", "yup", "it does", "leaves a dent", "stays indented", "pitting", "imprint remains"],
    "collateral_veins": ["yes", "yeah", "yep", "yup", "i have", "i see", "new veins", "visible veins", "popped up", "spidery veins"],
    "chf_history": ["yes", "yeah", "yep", "yup", "i have", "i was diagnosed", "i do", "heart failure", "chf", "my heart is weak", "congestive failure", "fluid in my lungs"],
    "hypertension": ["yes", "yeah", "yep", "yup", "i have", "i was diagnosed", "i do", "high blood pressure", "bp runs high", "lisinopril", "doctor says bp is high"],
    "stroke_tia_history": ["yes", "yeah", "yep", "yup", "i have", "i had", "stroke", "tia", "mini-stroke"],
    "vascular_disease": ["yes", "yeah", "yep", "yup", "i have", "i had", "heart attack", "mi", "pad", "stent", "bypass"],
    "diabetes": ["yes", "yeah", "yep", "yup", "i have", "i am", "diabetic", "diabetes", "type 1", "type 2", "sugar runs high"],
    "confusion": ["yes", "yeah", "yep", "yup", "i have", "i am", "i feel", "confused", "disoriented", "out of it", "not making sense", "muddled"],
    "urea_elevated": ["yes", "yeah", "yep", "yup", "i have", "it was high", "high urea", "high bun", "urea was high", "bun was high", "above normal"],
    "rr_high": ["yes", "yeah", "yep", "yup", "i am", "breathing fast", "fast breathing", "breathing quickly", "panting", "gasping", "short of breath", "breathless"],
    "bp_low": ["yes", "yeah", "yep", "yup", "it was low", "it is low", "low blood pressure", "bp was low", "bp is low", "measured low"],
}

NO_PATTERNS = {
    "fever": ["no", "nope", "not really", "i don't", "i do not", "no fever", "i'm not", "i am not"],
    "absence_of_cough": ["yes cough", "i have a cough", "i do cough", "coughing", "i'm coughing", "i am coughing"],
    "tender_cervical_nodes": ["no", "nope", "not really", "i don't", "i do not", "no lumps", "no glands", "not swollen"],
    "tonsillar_exudate": ["no", "nope", "not really", "i don't", "i do not", "no white", "no patches", "no swelling", "look normal", "looks normal", "look fine", "looks fine"],
    "active_cancer": ["no", "nope", "not really", "i don't", "i do not", "no cancer", "never", "cancer free"],
    "paralysis_or_immobilization": ["no", "nope", "not really", "i don't", "i do not", "no paralysis", "no cast", "no splint", "can walk fine", "full movement"],
    "bedridden_or_surgery": ["no", "nope", "not really", "i don't", "i do not", "no surgery", "no bedridden", "ambulatory"],
    "localized_tenderness": ["no", "nope", "not really", "i don't", "i do not", "not tender", "doesn't hurt to touch", "painless"],
    "entire_leg_swollen": ["no", "nope", "not really", "i don't", "i do not", "not the whole leg", "just the calf", "just the ankle", "localized swelling"],
    "calf_swelling_over_3cm": ["no", "nope", "not really", "i don't", "i do not", "not bigger", "same size"],
    "pitting_edema": ["no", "nope", "not really", "i don't", "i do not", "no dent", "doesn't leave a dent", "springs back"],
    "collateral_veins": ["no", "nope", "not really", "i don't", "i do not", "no new veins", "no visible veins", "same as always"],
    "chf_history": ["no", "nope", "not really", "i don't", "i do not", "no heart failure", "no chf", "heart is fine"],
    "hypertension": ["no", "nope", "not really", "i don't", "i do not", "no high blood pressure", "no hypertension", "bp is fine", "normal blood pressure"],
    "stroke_tia_history": ["no", "nope", "not really", "i don't", "i do not", "no stroke", "no tia", "never had a stroke"],
    "vascular_disease": ["no", "nope", "not really", "i don't", "i do not", "no heart attack", "no pad", "no vascular", "arteries are clear"],
    "diabetes": ["no", "nope", "not really", "i don't", "i do not", "no diabetes", "not diabetic", "sugar normal"],
    "confusion": ["no", "nope", "not really", "i don't", "i do not", "no confusion", "not confused", "clear-headed", "thinking clearly", "same as usual"],
    "urea_elevated": ["no", "nope", "not really", "i don't", "i do not", "no blood test", "haven't had blood", "not tested", "was normal", "levels were normal"],
    "rr_high": ["no", "nope", "not really", "i don't", "i do not", "breathing fine", "breathing normally", "normal breathing", "not breathless"],
    "bp_low": ["no", "nope", "not really", "i don't", "i do not", "bp is fine", "blood pressure is fine", "was normal", "not low", "no low"],
}


# ---- Extraction result type ----
# Returns either a value (bool/int/str) or the special string "unclear".
ExtractionValue = Union[bool, int, str, None]


def _resolve_yes_no(field: str, text: str, allow_unclear: bool = True) -> ExtractionValue:
    """
    Use the explicit YES/NO pattern tables to classify a short answer.

    Returns True, False, or "unclear".
    """
    t = text.lower().strip()

    # Bare uncertainty always wins.
    if _is_hedged(t):
        return "unclear"

    # "Ruled out" family — phrase-aware, checked before the single-token
    # NEGATORS fallback and the NO/YES pattern tables. Double-negation forms
    # first (more specific; they contain "ruled out" as a substring).
    for p in RULED_OUT_POSSIBLE:
        if p in t:
            return "unclear" if allow_unclear else None
    for p in RULED_OUT_NEGATED:
        if p in t:
            return True if field in INVERTED_FIELDS else False

    # Use word-boundary aware matching so "i do" doesn't match inside "doubt"
    # and "no" doesn't match inside "not". Use \b so punctuation (comma,
    # period) doesn't block matching at sentence boundaries.
    def _pattern_match(pattern: str, target: str) -> bool:
        return bool(re.search(rf"\b{re.escape(pattern)}\b", target))

    # Strong no markers first. For inverted-polarity fields (e.g.
    # absence_of_cough), NO_PATTERNS represent the underlying condition
    # (cough present). If such a pattern is negated ("not coughing"),
    # the condition is absent → criterion (absence) is True.
    # For normal fields, NO_PATTERNS are already explicit negative
    # expressions ("no", "never", "no cancer") and don't need negation
    # checking — they ARE the negation.
    for pat in NO_PATTERNS.get(field, []):
        if _pattern_match(pat, t):
            # Scope termination: if "but" separates this NO match from a
            # later affirmative clause for this field, the negation's scope
            # ends before that clause — skip this NO match and let the
            # YES_PATTERNS loop below evaluate the later clause instead.
            if _no_pattern_scope_ends_before_yes(t, pat, field):
                continue
            if field in INVERTED_FIELDS and _is_negated_context(t, pat):
                return True
            return False
    # Then yes markers — but check negation context first, so that
    # "never diagnosed with heart failure" doesn't return True just
    # because "heart failure" is a yes-pattern keyword. A negated
    # positive indicator is a clear negative (flip to False).
    # For inverted-polarity fields (e.g. absence_of_cough), a negated
    # positive keyword means the criterion IS present (flip to True).
    for pat in YES_PATTERNS.get(field, []):
        if _pattern_match(pat, t):
            if _is_negated_context(t, pat):
                return True if field in INVERTED_FIELDS else False
            return True

    # General yes/no detector as a last resort for very short answers.
    # For inverted-polarity fields (e.g. absence_of_cough), "yes" means the
    # underlying condition is present, so the criterion (absence) is False.
    # "no" means the condition is absent, so the criterion is True.
    if len(t.split()) <= 3:
        yn = detect_yes_no(t)
        if yn is True:
            return False if field in INVERTED_FIELDS else True
        if yn is False:
            return True if field in INVERTED_FIELDS else False

    if allow_unclear:
        return "unclear"
    return None


# ---- Leg swelling / Wells ----

WELLS_PATTERNS: Dict[str, List[str]] = {
    "active_cancer": ["cancer", "chemo", "chemotherapy", "tumor", "tumour", "on treatment", "radiation", "radiotherapy"],
    "paralysis_or_immobilization": ["paraly", "cast", "immobiliz", "weakness in", "can't move", "cannot move", "splint"],
    "bedridden_or_surgery": ["bedridden", "bed ridden", "surgery", "operation", "operated", "hospital", "bed for"],
    "localized_tenderness": ["tender", "sore calf", "sore leg", "painful to touch", "hurts to touch", "press on"],
    "entire_leg_swollen": ["whole leg", "entire leg", "all of the leg", "full leg", "both lower and upper"],
    "calf_swelling_over_3cm": ["bigger calf", "calf is bigger", "one calf", "3cm", "3 cm", "bigger than", "larger than", "swollen than", "asymmetric", "more swollen"],
    "pitting_edema": ["pitting", "dent", "leaves a dent", "stays indented", "finger press"],
    "collateral_veins": ["surface veins", "new veins", "visible veins", "collateral", "spider veins"],
}

WELLS_QUESTIONS = {
    "active_cancer": "Are you currently being treated for cancer, or have you been treated for cancer in the last 6 months?",
    "paralysis_or_immobilization": "Do you have any paralysis, leg weakness, or have you had a cast or splint on that leg recently?",
    "bedridden_or_surgery": "Have you been bedridden for more than 3 days recently, or had any major surgery in the past 12 weeks?",
    "localized_tenderness": "Is the swollen area tender or painful when you press on it, especially along the inner calf or thigh?",
    "entire_leg_swollen": "Is the whole leg swollen, or just one part like the calf or ankle?",
    "calf_swelling_over_3cm": "Is the calf of the swollen leg noticeably bigger than the other side — more than about 3 centimeters?",
    "pitting_edema": "If you press your finger into the swelling on that leg, does it leave a dent that stays for a moment?",
    "collateral_veins": "Have you noticed any new visible surface veins on that leg that weren't there before?",
}

# Static, pre-written rephrasings used when a patient answers "not sure".
# Same principle as the explanation templates: fixed human-authored text,
# never generated. Strategy per field is either (a) add a concrete example,
# or (b) split a compound question into the single most decisive part —
# bundling paralysis OR weakness OR cast into one question is a likely
# reason patients answer "not sure" in the first place.
WELLS_REPHRASINGS = {
    "active_cancer": "Let me ask that more simply: in the last 6 months, have you had chemotherapy, radiation, or surgery for cancer?",
    "paralysis_or_immobilization": "Let me break that into one simpler question: is that leg currently in a cast, brace, or splint — or unable to move normally?",
    "bedridden_or_surgery": "Let me simplify: in the last 3 months, have you had an operation of any kind, or spent more than 3 days in a row in bed?",
    "localized_tenderness": "To put it another way: if you press your fingers firmly into the swollen part of your calf or thigh, does it hurt?",
    "entire_leg_swollen": "Another way to ask: is the swelling only in the calf or ankle, or does it go all the way up to your thigh as well?",
    "calf_swelling_over_3cm": "Let me make that more concrete: if you look at both calves side by side, does the swollen one look clearly bigger — roughly an inch or more?",
    "pitting_edema": "To be more specific: press your thumb into the swollen area for about 5 seconds, then lift it off. Does a dent stay behind for a moment?",
    "collateral_veins": "Another way to put it: looking at the skin of that leg, do you see any veins near the surface that are new — ones you don't remember being there before?",
}

# Concrete tappable answer options, offered ONLY alongside the rephrased
# question (escalation attempt 2) — never on the first ask, where free text
# stays the default. Hand-written per field, same principle as
# WELLS_REPHRASINGS.
#
# IMPORTANT: a tapped chip is submitted as ordinary text and travels the
# normal _resolve_yes_no() path, exactly like a typed answer. Each label is
# therefore worded so the existing YES_PATTERNS / NO_PATTERNS tables resolve
# it correctly — there is deliberately no separate chip-handling branch.
# test_option_chips.py asserts every label round-trips to the intended value.
WELLS_OPTIONS = {
    "active_cancer": ["Yes, in the last 6 months", "No, never", "Not sure"],
    "paralysis_or_immobilization": ["Yes, I have a cast or brace", "No, I can move normally", "Not sure"],
    "bedridden_or_surgery": ["Yes, surgery or long bed rest", "No, neither of those", "Not sure"],
    "localized_tenderness": ["Yes, it hurts to press", "No, it doesn't hurt", "Not sure"],
    "entire_leg_swollen": ["The whole leg", "Just the calf or ankle", "Not sure"],
    "calf_swelling_over_3cm": ["Yes, clearly bigger", "No, about the same size", "Not sure"],
    "pitting_edema": ["Yes, it leaves a dent", "No, it springs back", "Not sure"],
    "collateral_veins": ["Yes, there are new veins", "No, nothing new", "Not sure"],
}

# Human-readable display names, used when a field has to be named in
# patient- or doctor-facing output (e.g. criteria left unresolved after
# repeated "not sure" answers). Static lookup table, same pattern as
# WELLS_QUESTIONS — never generated.
WELLS_FIELD_LABELS = {
    "active_cancer": "Active cancer or recent cancer treatment",
    "paralysis_or_immobilization": "Paralysis or leg immobilization",
    "bedridden_or_surgery": "Recent bed rest or major surgery",
    "localized_tenderness": "Tenderness along the deep leg veins",
    "entire_leg_swollen": "Swelling of the entire leg",
    "calf_swelling_over_3cm": "Calf more than 3 cm larger than the other side",
    "pitting_edema": "Pitting edema in the affected leg",
    "collateral_veins": "New surface (collateral) veins",
}


def extract_wells_fields(combined: str, current_input: str, missing: List[str], last_asked_field: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """Return (extracted_values, unclear_fields).

    Uses three signals in order of priority:
    1. If this is the last_asked_field, use explicit yes/no pattern tables.
    2. Descriptive keyword matches (e.g., "tender" in the opening description).
    3. General yes/no detection for very short answers.

    If none match, the field is marked "unclear" so the caller can prompt again
    with quick-reply chips.
    """
    out: Dict = {}
    unclear: List[str] = []
    cur_lower = current_input.lower()
    cur_yn = detect_yes_no(current_input)
    cur_is_no = cur_yn is False
    cur_short = len(cur_lower.split()) <= 6

    for k in missing:
        pats = WELLS_PATTERNS[k]
        cur_hit = _has_any(current_input, pats)
        combined_hit = _has_any(combined, pats)

        # Priority 1: answer to the field we just asked about.
        # Explicit yes/no patterns take precedence over descriptive keyword matches
        # so negated descriptions like "not diabetic" don't become True.
        if last_asked_field == k:
            resolved = _resolve_yes_no(k, current_input)
            if resolved is True:
                out[k] = True
                continue
            elif resolved is False:
                out[k] = False
                continue
            # Hedged input must never resolve to a confident True.
            if _is_hedged(current_input):
                unclear.append(k)
                continue
            # Descriptive keyword match — but only if not negated.
            if _positive_keyword_hit(current_input, pats):
                out[k] = True
                continue
            unclear.append(k)
            continue

        # Priority 2: descriptive keyword match anywhere in the conversation -> True
        # Negation-aware: "never had a TIA" must not become True.
        # Guard: only fire when current input is relevant to this field (cur_hit)
        # or is a bare yes/no answer (cur_yn is not None) — prevents greedy false
        # positives from keywords in earlier turns (e.g. "stroke" in opening msg).
        if _positive_keyword_hit(combined, pats) and not cur_is_no:
            if cur_hit or (cur_short and cur_yn is not None):
                out[k] = True
                continue

        # Priority 3: short yes/no answer when we don't know the topic
        if cur_short and cur_yn is not None and not cur_hit:
            # Don't guess which field a bare yes/no refers to unless we asked it
            unclear.append(k)
            continue

        # No signal
        unclear.append(k)

    return out, unclear


# ---- Sore throat / Centor ----

CENTOR_PATTERNS: Dict[str, List[str]] = {
    "fever": ["fever", "feverish", "temperature", "hot", "chills", "shivering", "burning up", "38 ", "100.4", "100 ", "101", "102", "103"],
    "absence_of_cough": ["no cough", "don't have a cough", "do not have a cough", "without cough", "isn't a cough", "not coughing"],
    "tender_cervical_nodes": ["swollen neck", "tender neck", "lymph node", "lumps in neck", "lumps on neck", "lumps on my neck", "lumps on the neck", "glands swollen", "swollen glands", "tender glands", "neck glands", "sore neck", "swollen lumps", "tender lumps", "lumps near", "lumps at the front"],
    "tonsillar_exudate": ["white patch", "white spots", "white coating", "pus on", "patches on", "patches on my", "white stuff", "swollen tonsils", "tonsils swollen", "exudate", "white on"],
}

COUGH_PRESENT = ["cough", "coughing"]

CENTOR_QUESTIONS = {
    "age": "How old are you?",
    "fever": "Have you had a fever — a temperature over 100.4 degrees Fahrenheit, or felt hot and shivery?",
    "absence_of_cough": "Do you have a cough along with your sore throat?",
    "tender_cervical_nodes": "Are there any tender or swollen lumps at the front of your neck, near the jaw?",
    "tonsillar_exudate": "If you look in the back of your throat, do you see any white patches or swelling on your tonsils?",
}

# Static, pre-written rephrasings for "not sure" answers (see WELLS_REPHRASINGS).
CENTOR_REPHRASINGS = {
    "age": "Just the number is fine — for example, 34.",
    "fever": "Let me simplify: have you measured your temperature and seen it above 100.4°F (38°C) — or, without measuring, felt hot and shivery?",
    "absence_of_cough": "Let me ask that more simply: have you been coughing at all over the last day or two?",
    "tender_cervical_nodes": "To be more specific: press gently along the front of your neck, just under your jawline. Do you feel any lumps there, or is it sore to press?",
    "tonsillar_exudate": "Another way to ask: with a light and a mirror, look at the back of your throat. Do your tonsils have white or yellow spots on them, or look puffy?",
}

# Escalation-only answer options (see WELLS_OPTIONS). `age` is intentionally
# absent: it is a free numeric value, so a short option list would either
# be wrong or force the patient into a bucket.
#
# absence_of_cough is inverted-polarity — the rephrased question asks whether
# the patient IS coughing, so "Yes, I have a cough" must resolve the
# criterion (absence of cough) to False. The labels are worded to hit the
# inverted-field branch of _resolve_yes_no() correctly.
CENTOR_OPTIONS = {
    "fever": ["Yes, felt hot or feverish", "No fever", "Not sure"],
    "absence_of_cough": ["Yes, I have a cough", "No, I'm not coughing", "Not sure"],
    "tender_cervical_nodes": ["Yes, tender lumps there", "No lumps I can feel", "Not sure"],
    "tonsillar_exudate": ["Yes, white patches", "No, they look normal", "Not sure / can't tell"],
}

# Human-readable display names (see WELLS_FIELD_LABELS).
CENTOR_FIELD_LABELS = {
    "age": "Age",
    "fever": "Fever",
    "absence_of_cough": "Absence of a cough",
    "tender_cervical_nodes": "Tender or swollen lymph nodes at the front of the neck",
    "tonsillar_exudate": "White patches or swelling on the tonsils",
}


def extract_centor_fields(combined: str, current_input: str, missing: List[str], last_asked_field: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """Return (extracted_values, unclear_fields)."""
    out: Dict = {}
    unclear: List[str] = []
    cur_lower = current_input.lower()
    cur_yn = detect_yes_no(current_input)
    cur_is_no = cur_yn is False
    cur_short = len(cur_lower.split()) <= 6

    for k in missing:
        if k == "age":
            a = extract_age(combined)
            if a is not None:
                out["age"] = a
            else:
                unclear.append("age")
            continue

        if k == "absence_of_cough":
            # Inverted-polarity field: "no cough" = True, "cough present" = False.
            # Try explicit yes/no resolution first (uses YES_PATTERNS which
            # include "haven't been coughing", "cough free", etc.), then fall
            # back to negation-aware cough-presence detection.
            resolved = _resolve_yes_no("absence_of_cough", current_input)
            if resolved is True:
                out["absence_of_cough"] = True
                continue
            elif resolved is False:
                out["absence_of_cough"] = False
                continue
            # Hedged input like "a tiny cough now and then" -> unclear.
            if _is_hedged(current_input):
                unclear.append("absence_of_cough")
                continue
            # Descriptive keyword match for cough presence (negation-aware
            # so "I haven't coughed" doesn't count as cough-present).
            if _positive_keyword_hit(current_input, COUGH_PRESENT):
                out["absence_of_cough"] = False
            elif _positive_keyword_hit(combined, COUGH_PRESENT) and not cur_is_no:
                out["absence_of_cough"] = False
            else:
                unclear.append("absence_of_cough")
            continue

        # Last-asked field: explicit yes/no patterns take precedence, then
        # descriptive keyword matches (e.g., "my tonsils look normal" -> False).
        if last_asked_field == k:
            pats_temp = CENTOR_PATTERNS[k]
            resolved = _resolve_yes_no(k, current_input)
            if resolved is True:
                out[k] = True
                continue
            elif resolved is False:
                out[k] = False
                continue
            # Hedged input must never resolve to a confident True.
            if _is_hedged(current_input):
                unclear.append(k)
                continue
            # Descriptive keyword match — but only if not negated.
            if _positive_keyword_hit(current_input, pats_temp):
                out[k] = True
                continue
            unclear.append(k)
            continue

        pats = CENTOR_PATTERNS[k]
        cur_hit = _has_any(current_input, pats)
        combined_hit = _has_any(combined, pats)

        # Descriptive keyword match -> True (negation-aware).
        # Guard: only fire when current input is relevant to this field (cur_hit)
        # or is a bare yes/no answer (cur_yn is not None) — prevents greedy false
        # positives from keywords in earlier turns.
        if _positive_keyword_hit(combined, pats) and not cur_is_no:
            if cur_hit or (cur_short and cur_yn is not None):
                out[k] = True
                continue

        # No signal
        if cur_short and cur_yn is not None and not cur_hit:
            unclear.append(k)
            continue
        unclear.append(k)

    return out, unclear


# ---- AFib / CHA₂DS₂-VASc ----

CHADSVASC_PATTERNS: Dict[str, List[str]] = {
    "chf_history": ["heart failure", "congestive heart failure", "chf", "weak heart", "heart can't pump", "pump failure"],
    "hypertension": ["high blood pressure", "hypertension", "blood pressure", "hbp", "bp"],
    "stroke_tia_history": ["stroke", "tia", "mini-stroke", "mini stroke", "transient ischemic", "blood clot in brain", "thromboembolism", "embolism"],
    "vascular_disease": ["heart attack", "mi", "myocardial infarction", "peripheral artery", "pad", "aortic plaque", "aortic", "blocked artery", "coronary", "stent", "bypass"],
    "diabetes": ["diabetes", "diabetic", "blood sugar", "high blood sugar", "type 1", "type 2", "insulin", "metformin", "glucose"],
}

CHADSVASC_QUESTIONS = {
    "age": "How old are you?",
    "sex": "Are you male or female?",
    "chf_history": "Have you ever been diagnosed with heart failure — sometimes called congestive heart failure?",
    "hypertension": "Do you have high blood pressure, or have you been told you have hypertension?",
    "stroke_tia_history": "Have you ever had a stroke, a mini-stroke (TIA), or any kind of blood clot?",
    "vascular_disease": "Have you ever had a heart attack, peripheral artery disease, or any vascular disease?",
    "diabetes": "Do you have diabetes?",
}

# Static, pre-written rephrasings for "not sure" answers (see WELLS_REPHRASINGS).
CHADSVASC_REPHRASINGS = {
    "age": "Just the number is fine — for example, 68.",
    "sex": "This score uses sex as one of its factors. Male or female is fine.",
    "chf_history": "Let me simplify: has a doctor ever told you your heart doesn't pump strongly enough, or put you on medication for heart failure?",
    "hypertension": "Another way to ask: has a doctor ever told you your blood pressure was high, or started you on blood pressure medication?",
    "stroke_tia_history": "Let me break that up and ask just the first part: have you ever had a stroke, or a 'mini-stroke' — also called a TIA?",
    "vascular_disease": "Let me simplify: have you ever had a heart attack, a stent or bypass surgery, or been told you have blocked arteries in your legs?",
    "diabetes": "Another way to ask: has a doctor ever told you that you have diabetes, or put you on insulin or metformin?",
}

# Escalation-only answer options (see WELLS_OPTIONS). `age` is omitted for
# the same reason as in CENTOR_OPTIONS (free numeric value).
CHADSVASC_OPTIONS = {
    "sex": ["Male", "Female"],
    "chf_history": ["Yes, diagnosed with heart failure", "No, never diagnosed", "Not sure"],
    "hypertension": ["Yes, or I take BP medication", "No, my BP is fine", "Not sure"],
    "stroke_tia_history": ["Yes, a stroke or TIA", "No, neither", "Not sure"],
    "vascular_disease": ["Yes, heart attack or stent", "No, none of those", "Not sure"],
    "diabetes": ["Yes, I have diabetes", "No, not diabetic", "Not sure"],
}

# Human-readable display names (see WELLS_FIELD_LABELS).
CHADSVASC_FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "chf_history": "History of heart failure",
    "hypertension": "High blood pressure",
    "stroke_tia_history": "Prior stroke or mini-stroke",
    "vascular_disease": "Vascular disease such as heart attack, blocked arteries, stent, or bypass",
    "diabetes": "Diabetes",
}


# ---- Pneumonia / CURB-65 ----
#
# Urea and blood pressure are measurements rather than things every patient
# knows. Patients who answer "no" or "not sure" to them are scored as
# not-elevated per the pathway spec — the report flags the tests as pending
# so the score reads as a floor, not a ceiling.

CURB65_PATTERNS: Dict[str, List[str]] = {
    "confusion": ["confused", "disoriented", "disorientated", "not making sense", "out of it", "muddled", "delirious"],
    "urea_elevated": ["urea", "bun", "blood urea", "high urea", "high bun", "kidney levels"],
    "rr_high": ["breathing fast", "breathing quickly", "breaths per minute", "breaths a minute", "short of breath", "shortness of breath", "breathless", "panting", "gasping", "breathing hard"],
    "bp_low": ["low blood pressure", "blood pressure was low", "blood pressure is low", "bp was low", "bp is low", "systolic", "diastolic", "pressure dropped"],
    "age_65_plus": ["65 or older", "over 65", "over sixty-five", "i'm 6", "i'm 7", "i'm 8", "i'm 9", "i am 6", "i am 7", "i am 8", "i am 9"],
}

CURB65_QUESTIONS = {
    "confusion": "Have you felt confused or disoriented recently, more than usual?",
    "urea_elevated": "Have you had a blood test showing high urea or BUN levels?",
    "rr_high": "Are you breathing faster than 30 breaths per minute, or feeling short of breath?",
    "bp_low": "Has your blood pressure been measured as low recently — systolic under 90 or diastolic under 60?",
    "age_65_plus": "Are you 65 years old or older?",
}

# Static, pre-written rephrasings for "not sure" answers (see WELLS_REPHRASINGS).
CURB65_REPHRASINGS = {
    "confusion": "Let me put that more simply: over the last day or two, have you had trouble thinking clearly, knowing where you are, or answering simple questions — more than is normal for you?",
    "urea_elevated": "Another way to ask: when you last had blood taken, did anyone mention your urea or kidney levels were high? If you haven't had blood tests recently, just say no.",
    "rr_high": "To be more concrete: sit quietly for a moment and count how many breaths you take in 30 seconds. More than 15 in half a minute means yes — otherwise no.",
    "bp_low": "Let me simplify: if your blood pressure was checked recently, did the person checking it say it was low? If it hasn't been checked, just say no.",
    "age_65_plus": "Just a yes or no is fine: are you aged 65 or older?",
}

# Escalation-only answer options (see WELLS_OPTIONS).
CURB65_OPTIONS = {
    "confusion": ["Yes, more confused than usual", "No, thinking clearly", "Not sure"],
    "urea_elevated": ["Yes, blood test showed high levels", "No, or haven't been tested", "Not sure"],
    "rr_high": ["Yes, breathing fast or short of breath", "No, breathing is normal", "Not sure"],
    "bp_low": ["Yes, it was measured low", "No, or it hasn't been checked", "Not sure"],
    "age_65_plus": ["Yes, I'm 65 or older", "No, I'm under 65", "Not sure"],
}

# Human-readable display names (see WELLS_FIELD_LABELS).
CURB65_FIELD_LABELS = {
    "confusion": "New confusion or disorientation",
    "urea_elevated": "High urea or BUN on blood test",
    "rr_high": "Fast breathing — more than 30 breaths per minute",
    "bp_low": "Low blood pressure",
    "age_65_plus": "Age 65 or older",
}


def _extract_rr_number(text: str) -> Optional[bool]:
    """Parse an explicit breaths-per-minute count ("about 35 per minute").
    >=30 -> True, <30 -> False, no number -> None."""
    m = re.search(r"(\d{2,3})\s*(?:per minute|a minute|/min|bpm|breaths)", text.lower())
    if not m:
        return None
    return int(m.group(1)) >= 30


def _extract_age_65_plus(text: str) -> Optional[bool]:
    """Parse an explicit age ("I'm 70", or a bare "70") into the >=65 boolean.
    None if absent."""
    a = extract_age(text)
    if a is None:
        m = re.fullmatch(r"\s*(\d{1,3})\s*", text)
        if m:
            a = int(m.group(1))
    if a is None:
        return None
    return a >= 65


def extract_curb65_fields(combined: str, current_input: str, missing: List[str], last_asked_field: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """Return (extracted_values, unclear_fields) for CURB-65."""
    out: Dict = {}
    unclear: List[str] = []
    cur_lower = current_input.lower()
    cur_yn = detect_yes_no(current_input)
    cur_is_no = cur_yn is False
    cur_short = len(cur_lower.split()) <= 6

    for k in missing:
        # Numeric/structured fields first.
        if k == "age_65_plus":
            # Current input first ("70" or "I'm 70" answering the question),
            # then any age mentioned earlier in the conversation.
            a = _extract_age_65_plus(current_input)
            if a is None:
                a = _extract_age_65_plus(combined)
            if a is not None:
                out["age_65_plus"] = a
                continue
            if last_asked_field == k:
                resolved = _resolve_yes_no(k, current_input)
                if resolved is True:
                    out[k] = True
                    continue
                elif resolved is False:
                    out[k] = False
                    continue
                if _is_hedged(current_input):
                    unclear.append(k)
                    continue
            unclear.append(k)
            continue

        if k == "rr_high":
            rr = _extract_rr_number(current_input) or _extract_rr_number(combined)
            if rr is not None:
                out["rr_high"] = rr
                continue

        # Last-asked field: explicit yes/no patterns take precedence, then
        # descriptive keyword matches (so negated descriptions don't become True).
        if last_asked_field == k:
            pats_temp = CURB65_PATTERNS[k]
            resolved = _resolve_yes_no(k, current_input)
            if resolved is True:
                out[k] = True
                continue
            elif resolved is False:
                out[k] = False
                continue
            # Hedged input must never resolve to a confident True.
            if _is_hedged(current_input):
                unclear.append(k)
                continue
            # Descriptive keyword match — but only if not negated.
            if _positive_keyword_hit(current_input, pats_temp):
                out[k] = True
                continue
            unclear.append(k)
            continue

        pats = CURB65_PATTERNS[k]
        cur_hit = _has_any(current_input, pats)
        combined_hit = _has_any(combined, pats)

        # Descriptive keyword match -> True (negation-aware).
        # Guard: only fire when current input is relevant to this field (cur_hit)
        # or is a bare yes/no answer (cur_yn is not None) — prevents greedy false
        # positives from keywords in earlier turns.
        if _positive_keyword_hit(combined, pats) and not cur_is_no:
            if cur_hit or (cur_short and cur_yn is not None):
                out[k] = True
                continue

        # No signal
        if cur_short and cur_yn is not None and not cur_hit:
            unclear.append(k)
            continue
        unclear.append(k)

    return out, unclear


def extract_chadsvasc_fields(combined: str, current_input: str, missing: List[str], last_asked_field: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """Return (extracted_values, unclear_fields)."""
    out: Dict = {}
    unclear: List[str] = []
    cur_lower = current_input.lower()
    cur_yn = detect_yes_no(current_input)
    cur_is_no = cur_yn is False
    cur_short = len(cur_lower.split()) <= 6

    for k in missing:
        if k == "age":
            a = extract_age(combined)
            if a is not None:
                out["age"] = a
            else:
                unclear.append("age")
            continue

        if k == "sex":
            combined_lower = combined.lower()
            if any(w in combined_lower for w in ["female", "woman", "lady", "girl"]):
                out["sex"] = "female"
            elif any(w in combined_lower for w in ["male", "man", "guy", "boy"]):
                out["sex"] = "male"
            else:
                unclear.append("sex")
            continue

        # Last-asked field: explicit yes/no patterns take precedence, then
        # descriptive keyword matches (so negated descriptions don't become True).
        if last_asked_field == k:
            pats_temp = CHADSVASC_PATTERNS[k]
            resolved = _resolve_yes_no(k, current_input)
            if resolved is True:
                out[k] = True
                continue
            elif resolved is False:
                out[k] = False
                continue
            # Hedged input must never resolve to a confident True.
            if _is_hedged(current_input):
                unclear.append(k)
                continue
            # Family-history exclusion: "my father had a heart attack" is
            # NOT personal vascular disease. CHA₂DS₂-VASc scores personal
            # history only.
            if _is_family_history(current_input):
                unclear.append(k)
                continue
            # Descriptive keyword match — but only if not negated.
            if _positive_keyword_hit(current_input, pats_temp):
                out[k] = True
                continue
            unclear.append(k)
            continue

        pats = CHADSVASC_PATTERNS[k]
        cur_hit = _has_any(current_input, pats)
        combined_hit = _has_any(combined, pats)

        # Descriptive keyword match -> True (negation-aware, family-history-aware).
        # Guard: only fire when current input is relevant to this field (cur_hit)
        # or is a bare yes/no answer (cur_yn is not None) — prevents greedy false
        # positives from keywords in earlier turns (e.g. "stroke" in opening msg).
        if _positive_keyword_hit(combined, pats) and not cur_is_no and not _is_family_history(current_input):
            if cur_hit or (cur_short and cur_yn is not None):
                out[k] = True
                continue

        # No signal
        if cur_short and cur_yn is not None and not cur_hit:
            unclear.append(k)
            continue
        unclear.append(k)

    return out, unclear


# ---- Out-of-scope symptom flagging ----
#
# When a patient mentions a symptom that matches NO criterion pattern in the
# active scoring instrument, we must neither silently discard it nor try to
# work out what it might mean. This detector ONLY answers the question
# "did the patient say something symptom-shaped that we did not screen for?"
# It performs no classification, names no condition, and returns the
# patient's own words verbatim.

# Clause-level qualifying language: a first-person symptom report.
_OOS_QUALIFIER_PATTERNS = [
    r"\bi\s+(?:\w+\s+)?(?:have|had|has|get|got|feel|felt|notice|noticed|been|keep|keeps)\b",
    r"\bi'?ve\s+(?:\w+\s+)?(?:got|had|been|noticed|felt)\b",
    r"\bi'?m\s+(?:\w+\s+)?(?:feeling|getting|having|noticing)\b",
    r"\bmy\s+\w+\s+(?:is|are|was|were|feels?|felt|hurts?|aches?|has|have|keeps?)\b",
    r"\bthere'?s\s+(?:a|some|been)\b",
]

# Body-related nouns and sensation words. Deliberately broad: a false
# negative here just means we stay silent, which is the safe direction.
_OOS_BODY_NOUNS = [
    "head", "headache", "migraine", "face", "jaw", "tooth", "teeth", "gum",
    "eye", "eyes", "vision", "ear", "ears", "hearing", "nose", "mouth",
    "tongue", "chest", "breast", "rib", "stomach", "belly", "abdomen",
    "abdominal", "gut", "bowel", "stool", "bladder", "urine", "urinating",
    "kidney", "liver", "back", "spine", "shoulder", "arm", "arms", "elbow",
    "wrist", "hand", "hands", "finger", "fingers", "hip", "knee", "shin",
    "foot", "feet", "toe", "toes", "heel", "skin", "rash", "hair", "nail",
    "muscle", "joint", "joints", "nerve", "bone",
    "numbness", "numb", "tingling", "pins and needles", "burning",
    "itching", "itchy", "cramp", "cramping", "spasm", "twitch", "weakness",
    "dizzy", "dizziness", "lightheaded", "faint", "nausea", "nauseous",
    "vomiting", "diarrhea", "constipation", "heartburn", "indigestion",
    "bloating", "fatigue", "tired", "exhausted", "sleep", "insomnia",
    "appetite", "weight", "sweating", "night sweats", "bruising", "bleeding",
    "breathing", "breath", "wheeze", "wheezing", "balance", "memory",
    "anxiety", "mood",
]

# Category "core" vocabulary: if a clause contains any of these, it is about
# the thing we ARE assessing, so it must never be flagged as out of scope.
# This sits on top of the per-criterion pattern tables as a safety net, so
# in-scope phrasing we simply failed to pattern-match doesn't get flagged.
_OOS_IN_SCOPE_TERMS = {
    "leg_swelling": [
        "leg", "legs", "calf", "calves", "thigh", "swell", "swollen",
        "swelling", "clot", "dvt", "vein", "veins", "cast", "splint",
        "cancer", "surgery", "bedridden", "tender", "edema", "dent",
    ],
    "sore_throat": [
        "throat", "swallow", "swallowing", "tonsil", "tonsils", "strep",
        "hoarse", "voice", "fever", "temperature", "cough", "coughing",
        "gland", "glands", "lymph", "neck",
    ],
    "afib_stroke": [
        "heart", "afib", "a-fib", "atrial", "fibrillation", "palpitation",
        "palpitations", "irregular", "pulse", "rhythm", "stroke", "tia",
        "blood pressure", "hypertension", "diabetes", "diabetic",
        "blood thinner", "anticoagulant", "clot",
    ],
    "pneumonia": [
        "cough", "coughing", "fever", "feverish", "pneumonia", "chest infection",
        "lung", "lungs", "phlegm", "sputum", "breath", "breathing", "breathless",
        "chest", "mucus", "wheeze", "chills",
    ],
}

_CATEGORY_PATTERN_TABLES = {
    "leg_swelling": WELLS_PATTERNS,
    "sore_throat": CENTOR_PATTERNS,
    "afib_stroke": CHADSVASC_PATTERNS,
    "pneumonia": CURB65_PATTERNS,
}

# Clause separators only. "also" is deliberately NOT a separator — it is an
# adverb that commonly sits inside the qualifying phrase ("I also have..."),
# and splitting on it strips the "I have" that the qualifier patterns need.
_OOS_CLAUSE_SPLIT = re.compile(r"[.;!?\n]|,| \band\b | \bbut\b ", re.IGNORECASE)


def detect_out_of_scope_mentions(text: str, category: str) -> List[str]:
    """Return clauses that look like a symptom report but match no criterion
    pattern for the active scoring instrument.

    Returns the patient's own words verbatim. Makes no attempt to classify,
    name, or interpret what was mentioned — that is explicitly out of scope
    for this tool and for this function.
    """
    pattern_table = _CATEGORY_PATTERN_TABLES.get(category)
    if pattern_table is None:
        return []

    all_criterion_patterns = [p for pats in pattern_table.values() for p in pats]
    in_scope_terms = _OOS_IN_SCOPE_TERMS.get(category, [])

    found: List[str] = []
    for raw_clause in _OOS_CLAUSE_SPLIT.split(text):
        if raw_clause is None:
            continue
        clause = raw_clause.strip()
        if not clause or len(clause.split()) < 3:
            continue
        low = clause.lower()

        # Skip clauses that are a plain negative ("I don't have any numbness").
        if detect_yes_no(clause) is False:
            continue
        # Skip anything that matches a criterion we DO screen for.
        if _has_any(low, all_criterion_patterns):
            continue
        # Skip anything using the active category's core vocabulary.
        if any(re.search(rf"\b{re.escape(t)}\b", low) for t in in_scope_terms):
            continue
        # Must look like a first-person symptom report...
        if not any(re.search(p, low) for p in _OOS_QUALIFIER_PATTERNS):
            continue
        # ...about a body part or bodily sensation ("s?" so plurals match).
        if not any(re.search(rf"\b{re.escape(n)}s?\b", low) for n in _OOS_BODY_NOUNS):
            continue

        if clause not in found:
            found.append(clause)

    return found


_CATEGORY_FIELD_LABELS = {
    "leg_swelling": WELLS_FIELD_LABELS,
    "sore_throat": CENTOR_FIELD_LABELS,
    "afib_stroke": CHADSVASC_FIELD_LABELS,
    "pneumonia": CURB65_FIELD_LABELS,
}

_CATEGORY_OPTIONS = {
    "leg_swelling": WELLS_OPTIONS,
    "sore_throat": CENTOR_OPTIONS,
    "afib_stroke": CHADSVASC_OPTIONS,
    "pneumonia": CURB65_OPTIONS,
}


def get_escalation_options(category: str, field: str) -> Optional[List[str]]:
    """Concrete answer options for a field, for use with the rephrased
    question only.

    Returns None when the field has no option list (e.g. `age`, a free
    numeric value), so the caller falls back to its normal chip behaviour.
    """
    options = _CATEGORY_OPTIONS.get(category, {}).get(field)
    return list(options) if options else None


def format_unresolved_fields(fields: List[str], category: str) -> List[str]:
    """Turn raw field keys into readable display strings.

    Each label is suffixed with "(not established)" so the reader can see
    at a glance that the criterion was never answered — as opposed to
    having been answered "no", which is how a bare criterion name would
    otherwise read.

    Unknown keys fall back to a de-underscored version of the key rather
    than being dropped, so a field can never silently vanish from the
    report just because someone forgot to add a label.
    """
    labels = _CATEGORY_FIELD_LABELS.get(category, {})
    out: List[str] = []
    for field in fields:
        label = labels.get(field) or field.replace("_", " ").capitalize()
        out.append(f"{label} (not established)")
    return out


def format_out_of_scope_notes(mentions: List[str], category: str) -> Optional[str]:
    """Build the report section text for out-of-scope mentions.

    Deliberately says only "we did not screen for this". It must never
    suggest what the symptom could indicate.
    """
    if not mentions:
        return None
    instrument_names = {
        "leg_swelling": "DVT",
        "sore_throat": "strep throat",
        "afib_stroke": "AFib stroke risk",
    }
    name = instrument_names.get(category, "this assessment's")
    quoted = "; ".join(f'"{m}"' for m in mentions)
    return (
        f"Other things you mentioned that we didn't screen for: {quoted}. "
        f"These aren't part of the {name} criteria this tool checks — "
        f"worth mentioning to your doctor separately."
    )
