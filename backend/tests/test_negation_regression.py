"""
Negation regression tests for DEEPEN-style and NegEx-style failure modes.

These tests are intentionally written against the *intended* behavior described
in docs/negation-pattern-diff-proposal.md §4.3. Some currently FAIL against
the existing rule_fallback.py implementation; those failures document the
baseline gaps that Stage 2's phrase-aware negation helper is meant to close.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extraction.rule_fallback import (
    extract_centor_fields,
    extract_chadsvasc_fields,
    _is_negated_context,
    _positive_keyword_hit,
)


class TestNegationRegression:
    """
    Regression tests for phrase-aware negation, pseudo-negation, and scope
    termination. Each test targets a specific failure mode identified in the
    Stage 2 diff proposal.
    """

    # ------------------------------------------------------------
    # Pseudo-negation / long-distance negation (DEEPEN-style)
    # ------------------------------------------------------------

    def test_no_evidence_of_extension_does_not_negate_calf(self):
        """
        'No evidence of extension of the swollen calf into the thigh.'
        The negation applies to 'extension', not to the swollen calf itself.
        A keyword match on 'calf' should not be treated as negated.
        """
        assert _is_negated_context(
            "No evidence of extension of the swollen calf into the thigh.",
            "calf"
        ) is False

    def test_no_significant_change_does_not_negate_swelling(self):
        """
        'No significant change in my tonsillar swelling.'
        The negation applies to 'change', not to the swelling itself.
        A keyword match on 'swelling' should remain a positive signal.
        """
        assert _positive_keyword_hit(
            "No significant change in my tonsillar swelling.",
            ["swelling"]
        ) is True

    # ------------------------------------------------------------
    # Strong completed negation
    # ------------------------------------------------------------

    def test_ruled_out_stroke_is_false(self):
        """
        'The doctor ruled out a stroke, so it must be something else.'
        'ruled out' is a strong completed negation; stroke history should be False.
        """
        result, unclear = extract_chadsvasc_fields(
            "",
            "The doctor ruled out a stroke, so it must be something else.",
            ["stroke_tia_history"],
            last_asked_field="stroke_tia_history"
        )
        assert result.get("stroke_tia_history") is False
        assert "stroke_tia_history" not in unclear

    # ------------------------------------------------------------
    # Possible / double negation
    # ------------------------------------------------------------

    def test_not_been_ruled_out_is_unclear(self):
        """
        'A stroke has not been ruled out yet.'
        Negation of a negation: the condition remains possible, not confirmed absent.
        Should return unclear, not False.
        """
        result, unclear = extract_chadsvasc_fields(
            "",
            "A stroke has not been ruled out yet.",
            ["stroke_tia_history"],
            last_asked_field="stroke_tia_history"
        )
        assert result.get("stroke_tia_history") is None
        assert "stroke_tia_history" in unclear

    # ------------------------------------------------------------
    # Scope termination across conjunctions
    # ------------------------------------------------------------

    def test_but_preserves_fever_after_negated_cough(self):
        """
        'I have no cough, but I do have a fever.'
        The negation 'no' belongs to 'cough'; the conjunction 'but' should
        terminate the scope so 'fever' is not negated.
        """
        result, unclear = extract_centor_fields(
            "",
            "I have no cough, but I do have a fever.",
            ["fever"],
            last_asked_field="fever"
        )
        assert result.get("fever") is True
        assert "fever" not in unclear

    # ------------------------------------------------------------
    # Ordinary adjacent negation (corrected from a flawed earlier test)
    # ------------------------------------------------------------

    def test_not_present_directly_negates_blood_clot(self):
        """
        'The swelling is secondary to the sprain; blood clot symptoms are not present.'
        'not' is directly adjacent to 'blood clot symptoms' — this is ordinary
        adjacent negation, not a scope-crossing case. It should already be
        negated correctly WITHOUT any 'secondary to' scope-termination logic.

        (This replaces a previous version of this test that incorrectly
        asserted the opposite: the original sentence used "not a blood clot"
        immediately adjacent to the keyword, which is a real, correctly-scoped
        negation, not a crossing bug. See Stage 2c-fix.)
        """
        assert _is_negated_context(
            "The swelling is secondary to the sprain; blood clot symptoms are not present.",
            "blood clot"
        ) is True

    # ------------------------------------------------------------
    # Scope terminator prevents unrelated negation crossing
    # ------------------------------------------------------------

    def test_secondary_to_does_not_pull_earlier_concern_into_negation(self):
        """
        'Blood clot was mentioned as a concern, but the swelling is secondary
        to the sprain.'
        There is no negation trigger anywhere in this sentence, so 'blood
        clot' must NOT be treated as negated. This test documents that the
        current implementation already gets this right WITHOUT any dedicated
        'secondary to' scope-termination logic (there's nothing to terminate
        the scope of, since no negator exists here). If 'secondary to' scope
        termination is ever implemented, this test must continue to pass.
        """
        assert _is_negated_context(
            "Blood clot was mentioned as a concern, but the swelling is secondary to the sprain.",
            "blood clot"
        ) is False
        assert _positive_keyword_hit(
            "Blood clot was mentioned as a concern, but the swelling is secondary to the sprain.",
            ["blood clot"]
        ) is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
