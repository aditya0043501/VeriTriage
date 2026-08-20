"""
Probability mapping: computed score -> ONLY the published probability range.

Every number returned by this module traces to a specific published source,
recorded in docs/clinical-explanation-sources.md. If a score bracket has no
cited source, this module raises UncitedScoreError rather than guessing,
interpolating, or extrapolating.

Sources (full citations in docs/clinical-explanation-sources.md):
- Wells PS, Owen C, et al. "Does this patient have deep vein thrombosis?"
  JAMA. 2006;295(2):199-207. (pooled prevalence by Wells clinical probability)
- McIsaac WJ, et al. "Empirical validation of guidelines for the management
  of pharyngitis in children and adults." JAMA. 2004;291(13):1587-1595.
  (modified Centor score probability table)
"""

from dataclasses import dataclass


class UncitedScoreError(ValueError):
    """Raised when a score falls in a bracket we have no cited source for.

    The caller must surface this as 'no published probability available' —
    it must never be silently replaced with a guessed number.
    """


WELLS_2006_JAMA_CITATION = (
    "Wells PS, Owen C, Doucette S, Fergusson D, Tran H. "
    "Does this patient have deep vein thrombosis? JAMA. 2006;295(2):199-207."
)
MCISAAC_2004_JAMA_CITATION = (
    "McIsaac WJ, Kellner JD, Aufricht P, Vanjaka A, Low DE. "
    "Empirical validation of guidelines for the management of pharyngitis "
    "in children and adults. JAMA. 2004;291(13):1587-1595."
)


@dataclass(frozen=True)
class ProbabilityResult:
    """A published probability bracket, with its source attached."""
    score_bracket: str        # e.g. "score <= 0"
    tier: str                 # "low" / "moderate" / "high"
    probability_text: str     # exactly as published, e.g. "5.0% (95% CI 4.0%-8.0%)"
    citation: str             # full citation string


# Mathematically possible score ranges given our scoring formulas.
# Anything outside these is malformed input, not a real score.
_WELLS_MIN, _WELLS_MAX = 0, 8       # 8 one-point criteria, no negative terms in our partial score
_CENTOR_MIN, _CENTOR_MAX = -1, 5    # 4 one-point criteria + age modifier (-1/0/+1)


def get_wells_probability(score: int) -> ProbabilityResult:
    """Map a computed Wells' DVT score to the published pooled DVT prevalence.

    Source: Wells 2006 JAMA systematic review (14 studies, >8,000 patients).
    Raises UncitedScoreError for malformed input.
    """
    if not isinstance(score, int) or isinstance(score, bool):
        raise UncitedScoreError(
            f"Wells score must be an integer, got {type(score).__name__}: {score!r}. "
            "No published probability can be cited for this input."
        )
    if score < _WELLS_MIN or score > _WELLS_MAX:
        raise UncitedScoreError(
            f"Wells score {score} is outside the possible range "
            f"[{_WELLS_MIN}, {_WELLS_MAX}] for our partial score. "
            "No published probability can be cited for this input."
        )

    if score <= 0:
        return ProbabilityResult(
            score_bracket="score <= 0",
            tier="low",
            probability_text="5.0% (95% CI 4.0%-8.0%)",
            citation=WELLS_2006_JAMA_CITATION,
        )
    elif score <= 2:
        return ProbabilityResult(
            score_bracket="score 1-2",
            tier="moderate",
            probability_text="17% (95% CI 13%-23%)",
            citation=WELLS_2006_JAMA_CITATION,
        )
    else:
        return ProbabilityResult(
            score_bracket="score >= 3",
            tier="high",
            probability_text="53% (95% CI 44%-61%)",
            citation=WELLS_2006_JAMA_CITATION,
        )


def get_centor_probability(score: int) -> ProbabilityResult:
    """Map a computed modified (McIsaac) Centor score to the published
    likelihood of streptococcal pharyngitis.

    Source: McIsaac 2004 JAMA validation (per-score-level table).
    Raises UncitedScoreError for malformed input or scores outside the
    mathematically possible range of the modified Centor score.
    """
    if not isinstance(score, int) or isinstance(score, bool):
        raise UncitedScoreError(
            f"Centor score must be an integer, got {type(score).__name__}: {score!r}. "
            "No published probability can be cited for this input."
        )
    if score < _CENTOR_MIN or score > _CENTOR_MAX:
        raise UncitedScoreError(
            f"Modified Centor score {score} is outside the possible range "
            f"[{_CENTOR_MIN}, {_CENTOR_MAX}]. "
            "No published probability can be cited for this input."
        )

    if score <= 0:
        return ProbabilityResult(
            score_bracket="score <= 0 (-1 or 0)",
            tier="low",
            probability_text="1%-2.5%",
            citation=MCISAAC_2004_JAMA_CITATION,
        )
    elif score == 1:
        return ProbabilityResult(
            score_bracket="score 1",
            tier="low",
            probability_text="5%-10%",
            citation=MCISAAC_2004_JAMA_CITATION,
        )
    elif score == 2:
        return ProbabilityResult(
            score_bracket="score 2",
            tier="moderate",
            probability_text="11%-17%",
            citation=MCISAAC_2004_JAMA_CITATION,
        )
    elif score == 3:
        return ProbabilityResult(
            score_bracket="score 3",
            tier="moderate",
            probability_text="28%-35%",
            citation=MCISAAC_2004_JAMA_CITATION,
        )
    else:  # 4 or 5
        return ProbabilityResult(
            score_bracket="score 4-5",
            tier="high",
            probability_text="51%-53%",
            citation=MCISAAC_2004_JAMA_CITATION,
        )
