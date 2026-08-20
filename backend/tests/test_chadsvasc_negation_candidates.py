"""
CHA₂DS₂-VASc-specific negation candidate tests (Stage 6, Part 2).

Companion to `test_negation_regression.py`, which covers Wells'/Centor.
These tests assert the *clinically correct* answer for each phrase, per
docs/chadsvasc-negation-audit.md. Some are EXPECTED TO FAIL against the
current implementation — those failures are the genuine gaps identified in
the audit, kept here as a baseline. No production code has been changed
in this pass; do not "fix" these by loosening the assertions.

Each test is tagged with the audit section it corresponds to and whether
it currently passes or fails, so a future fix pass has a fixed target list.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extraction.rule_fallback import extract_chadsvasc_fields


class TestChadsvascNegationCandidates:

    # ------------------------------------------------------------
    # §2.1 CHF — opportunistic-path "ruled out" gap
    # ------------------------------------------------------------

    def test_chf_ruled_out_last_asked_passes(self):
        """Baseline: this already works via _resolve_yes_no. (PASSES today)"""
        result, unclear = extract_chadsvasc_fields(
            "Heart failure was ruled out.",
            "Heart failure was ruled out.",
            ["chf_history"],
            last_asked_field="chf_history",
        )
        assert result.get("chf_history") is False

    def test_chf_ruled_out_opportunistic_fails_today(self):
        """
        Same fact, mentioned unprompted while a different field was asked.
        Genuine gap: opportunistic path has no 'ruled out' phrase check.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.1)
        """
        text = "By the way, heart failure was ruled out last year."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["chf_history", "hypertension"], last_asked_field="hypertension"
        )
        assert result.get("chf_history") is False

    # ------------------------------------------------------------
    # §2.2 Hypertension — "not ruled out" (works, accidentally) and "unlikely" (gap)
    # ------------------------------------------------------------

    def test_hypertension_not_ruled_out_last_asked_passes(self):
        """Baseline: double-negation via _resolve_yes_no. (PASSES today)"""
        result, unclear = extract_chadsvasc_fields(
            "Hypertension has not been ruled out.",
            "Hypertension has not been ruled out.",
            ["hypertension"],
            last_asked_field="hypertension",
        )
        assert result.get("hypertension") is None
        assert "hypertension" in unclear

    def test_hypertension_unlikely_fails_today(self):
        """
        'Unlikely' is a possible-negation term (diff proposal §2.4), never
        implemented for any module. Should route to unclear, not True.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.2)
        """
        text = "Hypertension is unlikely based on my last checkup."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["hypertension"], last_asked_field="hypertension"
        )
        assert result.get("hypertension") is None
        assert "hypertension" in unclear

    # ------------------------------------------------------------
    # §2.3 Stroke/TIA — opportunistic gap AND compound-field scope-term gap
    # ------------------------------------------------------------

    def test_stroke_ruled_out_last_asked_passes(self):
        """Baseline, same sentence as the existing Wells/Centor regression
        suite's test_ruled_out_stroke_is_false. (PASSES today)"""
        text = "The doctor ruled out a stroke, so it must be something else."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["stroke_tia_history"], last_asked_field="stroke_tia_history"
        )
        assert result.get("stroke_tia_history") is False

    def test_stroke_ruled_out_opportunistic_fails_today(self):
        """
        Identical sentence to the test above, but the field wasn't the one
        just asked about. Opposite (wrong) answer.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.3a)
        """
        text = "The doctor ruled out a stroke, so it must be something else."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["stroke_tia_history", "diabetes"], last_asked_field="diabetes"
        )
        assert result.get("stroke_tia_history") is False

    def test_no_stroke_but_tia_fails_today_even_on_last_asked_path(self):
        """
        The exact example phrase from the Stage 6 task. Confirmed broken
        even via the 'already fixed' last-asked path: 'stroke' and 'tia'
        are synonyms for the SAME compound field, and the existing `but`
        scope-termination only protects cross-field leakage, not
        same-field intra-clause conflicts.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.3b)
        """
        text = "No history of stroke, but I did have a TIA last year."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["stroke_tia_history"], last_asked_field="stroke_tia_history"
        )
        assert result.get("stroke_tia_history") is True

    def test_no_stroke_but_tia_opportunistic_fails_today(self):
        """
        Same phrase, opportunistic context — also wrong today, though it
        fails safe (unclear) rather than confidently wrong (False), because
        the blunt whole-message `cur_is_no` gate suppresses the entire
        opportunistic branch whenever any 'no' word appears anywhere in the
        message. (FAILS today — see docs/chadsvasc-negation-audit.md §2.3)
        """
        text = "No history of stroke, but I did have a TIA last year."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["stroke_tia_history", "diabetes"], last_asked_field="diabetes"
        )
        assert result.get("stroke_tia_history") is True

    def test_stroke_never_diagnosed_passes(self):
        """Baseline: single-token 'never' within the 4-word window already
        works correctly. (PASSES today)"""
        text = "Stroke was never diagnosed."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["stroke_tia_history"], last_asked_field="stroke_tia_history"
        )
        assert result.get("stroke_tia_history") is False

    # ------------------------------------------------------------
    # §2.4 Vascular disease — opportunistic gap + missing keyword coverage
    # ------------------------------------------------------------

    def test_vascular_disease_ruled_out_last_asked_passes(self):
        """Baseline. (PASSES today)"""
        text = "Vascular disease was ruled out after testing."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["vascular_disease"], last_asked_field="vascular_disease"
        )
        assert result.get("vascular_disease") is False

    def test_heart_attack_not_ruled_out_last_asked_passes(self):
        """Baseline: double-negation via _resolve_yes_no. (PASSES today)"""
        text = "A heart attack has not been ruled out yet."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["vascular_disease"], last_asked_field="vascular_disease"
        )
        assert result.get("vascular_disease") is None
        assert "vascular_disease" in unclear

    def test_no_heart_attack_but_diabetic_vascular_field_passes(self):
        """Baseline: cross-field 'but' scope-termination (vascular field
        correctly negated). (PASSES today)"""
        text = "No heart attacks or strokes for me, but I am diabetic."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["vascular_disease"], last_asked_field="vascular_disease"
        )
        assert result.get("vascular_disease") is False

    def test_no_heart_attack_but_diabetic_diabetes_field_passes(self):
        """Baseline: cross-field 'but' scope-termination (diabetes field
        correctly affirmed after 'but'). (PASSES today)"""
        text = "No heart attacks or strokes for me, but I am diabetic."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["diabetes"], last_asked_field="diabetes"
        )
        assert result.get("diabetes") is True

    def test_no_heart_attack_but_diabetic_diabetes_opportunistic_fails_today(self):
        """
        Same phrase, but 'diabetes' is not the last-asked field (some other
        field, e.g. hypertension, was asked instead, and diabetes is picked
        up opportunistically). Should still be True; the whole-message
        cur_is_no gate blocks it.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.3)
        """
        text = "No heart attacks or strokes for me, but I am diabetic."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["diabetes", "hypertension"], last_asked_field="hypertension"
        )
        assert result.get("diabetes") is True

    # ------------------------------------------------------------
    # §2.5 Diabetes — opportunistic "ruled out" gap + "unlikely" gap
    # ------------------------------------------------------------

    def test_diabetes_ruled_out_last_asked_passes(self):
        """Baseline. (PASSES today)"""
        text = "Diabetes was ruled out last year."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["diabetes"], last_asked_field="diabetes"
        )
        assert result.get("diabetes") is False

    def test_diabetes_ruled_out_opportunistic_fails_today(self):
        """
        Same gap as CHF (§2.1), applied to diabetes.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.5)
        """
        text = "Diabetes was ruled out last year, just so you know."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["diabetes", "hypertension"], last_asked_field="hypertension"
        )
        assert result.get("diabetes") is False

    def test_diabetes_unlikely_fails_today(self):
        """
        Same 'unlikely' gap as hypertension (§2.2), applied to diabetes.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.5)
        """
        text = "Diabetes is unlikely, according to my labs."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["diabetes"], last_asked_field="diabetes"
        )
        assert result.get("diabetes") is None
        assert "diabetes" in unclear

    # ------------------------------------------------------------
    # §2.6 Sex category — negation-blind substring match (related, not
    # part of the "ruled out" family, but same root cause)
    # ------------------------------------------------------------

    def test_sex_negation_not_a_woman_fails_today(self):
        """
        'not a woman' is matched as a bare 'woman'/'female' substring with
        no negation check, in both the sex branch of
        extract_chadsvasc_fields() and the near-duplicate logic in
        afib_extractor.py. Directly flips a scored point (+1 for female).
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.6)
        """
        text = "I'm not a woman, I'm a man."
        result, unclear = extract_chadsvasc_fields(text, text, ["sex"])
        assert result.get("sex") == "male"

    def test_sex_negation_not_a_woman_reordered_fails_today(self):
        """Same bug, negation stated after the male self-identification.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.6)"""
        text = "I am a man, not a woman."
        result, unclear = extract_chadsvasc_fields(text, text, ["sex"])
        assert result.get("sex") == "male"

    # ------------------------------------------------------------
    # §2.7 Family-history exclusion bypassed by direct YES_PATTERNS match
    # (bonus finding, not part of the negation-family scope, kept here for
    # traceability since it was found via the same audit)
    # ------------------------------------------------------------

    def test_family_history_chf_not_personal_fails_today(self):
        """
        CHA₂DS₂-VASc scores personal history only. '_resolve_yes_no()'
        matches the bare keyword 'heart failure' and returns True before
        the family-history guard (_is_family_history) ever runs.
        (FAILS today — see docs/chadsvasc-negation-audit.md §2.7)
        """
        text = "My father had heart failure."
        result, unclear = extract_chadsvasc_fields(
            text, text, ["chf_history"], last_asked_field="chf_history"
        )
        assert result.get("chf_history") is None
        assert "chf_history" in unclear


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
