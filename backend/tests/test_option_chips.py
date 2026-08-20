"""
Tests for escalation option chips.

Chips are offered ONLY with the rephrased question (escalation attempt 2).
Free text remains the default on the first ask and remains accepted at all
times.

The core guarantee under test: a tapped chip is submitted as ordinary text
and resolves through the SAME _resolve_yes_no() path as a typed answer.
There is no separate chip-handling branch, so these tests assert that each
chip label produces exactly the value the equivalent typed phrase would.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.rule_fallback import (  # noqa: E402
    WELLS_QUESTIONS,
    WELLS_OPTIONS,
    CENTOR_QUESTIONS,
    CENTOR_OPTIONS,
    CHADSVASC_QUESTIONS,
    CHADSVASC_OPTIONS,
    get_escalation_options,
    _resolve_yes_no,
    extract_wells_fields,
    extract_centor_fields,
    extract_chadsvasc_fields,
)
from extraction.extraction_utils import MAX_UNCLEAR_ATTEMPTS  # noqa: E402
from extraction.leg_swelling_extractor import (  # noqa: E402
    extract_and_update_data as wells_extract,
    LegSwellingData,
)
from main import _get_chips  # noqa: E402


# ---------------------------------------------------------------------------
# Chip labels must resolve identically to the equivalent typed answer
# ---------------------------------------------------------------------------

class TestChipsResolveLikeTypedText(unittest.TestCase):
    """Requirement 3: reuse the existing _resolve_yes_no() path.

    Each table is ordered [affirmative, negative, uncertain]. Chips are
    plain text, so if these resolve correctly then a tap and a type are
    indistinguishable to the extractor by construction.
    """

    # absence_of_cough is inverted-polarity: the rephrased question asks
    # whether the patient IS coughing, so an affirmative answer means the
    # criterion (absence of cough) is False.
    EXPECTED = {
        "default": [True, False, "unclear"],
        "absence_of_cough": [False, True, "unclear"],
    }

    def _check_table(self, table, name):
        for field, options in table.items():
            if field == "sex":
                continue  # resolved by the sex branch, covered separately
            expected = self.EXPECTED.get(field, self.EXPECTED["default"])
            self.assertEqual(
                len(options), len(expected),
                f"{name}.{field} must have {len(expected)} options",
            )
            for label, want in zip(options, expected):
                with self.subTest(table=name, field=field, label=label):
                    self.assertEqual(_resolve_yes_no(field, label), want)

    def test_wells_chips_resolve_correctly(self):
        self._check_table(WELLS_OPTIONS, "WELLS_OPTIONS")

    def test_centor_chips_resolve_correctly(self):
        self._check_table(CENTOR_OPTIONS, "CENTOR_OPTIONS")

    def test_chadsvasc_chips_resolve_correctly(self):
        self._check_table(CHADSVASC_OPTIONS, "CHADSVASC_OPTIONS")


class TestChipEqualsTypedPhraseEndToEnd(unittest.TestCase):
    """Requirement 5: a tapped chip gives the same extraction result as the
    equivalent typed phrase, across all three instruments."""

    def test_wells_paralysis_chip_matches_typed(self):
        chip = WELLS_OPTIONS["paralysis_or_immobilization"][0]  # affirmative
        typed = "yes"
        field = "paralysis_or_immobilization"
        from_chip, _ = extract_wells_fields(chip, chip, [field], field)
        from_typed, _ = extract_wells_fields(typed, typed, [field], field)
        self.assertEqual(from_chip, from_typed)
        self.assertIs(from_chip[field], True)

    def test_wells_pitting_edema_negative_chip_matches_typed(self):
        chip = WELLS_OPTIONS["pitting_edema"][1]  # negative
        typed = "no"
        field = "pitting_edema"
        from_chip, _ = extract_wells_fields(chip, chip, [field], field)
        from_typed, _ = extract_wells_fields(typed, typed, [field], field)
        self.assertEqual(from_chip, from_typed)
        self.assertIs(from_chip[field], False)

    def test_centor_fever_chip_matches_typed(self):
        chip = CENTOR_OPTIONS["fever"][0]
        typed = "yes"
        field = "fever"
        from_chip, _ = extract_centor_fields(chip, chip, [field], field)
        from_typed, _ = extract_centor_fields(typed, typed, [field], field)
        self.assertEqual(from_chip, from_typed)
        self.assertIs(from_chip[field], True)

    def test_centor_inverted_cough_chip_matches_typed_equivalent(self):
        """'Yes, I have a cough' must equal typing 'I have a cough', NOT 'yes'."""
        chip = CENTOR_OPTIONS["absence_of_cough"][0]
        typed = "I have a cough"
        field = "absence_of_cough"
        from_chip, _ = extract_centor_fields(chip, chip, [field], field)
        from_typed, _ = extract_centor_fields(typed, typed, [field], field)
        self.assertEqual(from_chip, from_typed)
        self.assertIs(from_chip[field], False)

    def test_chadsvasc_diabetes_chip_matches_typed(self):
        chip = CHADSVASC_OPTIONS["diabetes"][0]
        typed = "yes"
        field = "diabetes"
        from_chip, _ = extract_chadsvasc_fields(chip, chip, [field], field)
        from_typed, _ = extract_chadsvasc_fields(typed, typed, [field], field)
        self.assertEqual(from_chip, from_typed)
        self.assertIs(from_chip[field], True)

    def test_chadsvasc_stroke_negative_chip_matches_typed(self):
        chip = CHADSVASC_OPTIONS["stroke_tia_history"][1]
        typed = "no"
        field = "stroke_tia_history"
        from_chip, _ = extract_chadsvasc_fields(chip, chip, [field], field)
        from_typed, _ = extract_chadsvasc_fields(typed, typed, [field], field)
        self.assertEqual(from_chip, from_typed)
        self.assertIs(from_chip[field], False)

    def test_chadsvasc_sex_chips_resolve(self):
        """'Female' must not be swallowed by the 'male' substring check."""
        for label, expected in [("Male", "male"), ("Female", "female")]:
            with self.subTest(label=label):
                got, _ = extract_chadsvasc_fields(label, label, ["sex"], "sex")
                self.assertEqual(got.get("sex"), expected)

    def test_uncertain_chip_leaves_field_unresolved(self):
        field = "collateral_veins"
        chip = WELLS_OPTIONS[field][2]  # "Not sure"
        got, unclear = extract_wells_fields(chip, chip, [field], field)
        self.assertNotIn(field, got)
        self.assertIn(field, unclear)


# ---------------------------------------------------------------------------
# Chips appear only at the escalation moment
# ---------------------------------------------------------------------------

class TestChipsOnlyOnEscalation(unittest.TestCase):
    """Requirement 1: free text is the default; chips are not attached to
    every question up front."""

    def _data(self, field, unclear_count):
        return {
            "last_asked_field": field,
            field: None,
            "unclear_counts": ({field: unclear_count} if unclear_count else {}),
        }

    def test_first_ask_gets_generic_chips_not_field_options(self):
        data = self._data("paralysis_or_immobilization", 0)
        chips = _get_chips("extraction", data, "leg_swelling")
        self.assertEqual(chips, ["Yes", "No", "Not sure"])
        self.assertNotEqual(chips, WELLS_OPTIONS["paralysis_or_immobilization"])

    def test_rephrase_moment_gets_field_specific_options(self):
        data = self._data("paralysis_or_immobilization", MAX_UNCLEAR_ATTEMPTS - 1)
        chips = _get_chips("extraction", data, "leg_swelling")
        self.assertEqual(chips, WELLS_OPTIONS["paralysis_or_immobilization"])

    def test_answered_field_does_not_get_escalation_options(self):
        data = {
            "last_asked_field": "paralysis_or_immobilization",
            "paralysis_or_immobilization": True,
            "unclear_counts": {"paralysis_or_immobilization": 1},
        }
        chips = _get_chips("extraction", data, "leg_swelling")
        self.assertNotEqual(chips, WELLS_OPTIONS["paralysis_or_immobilization"])

    def test_field_without_options_falls_back(self):
        """`age` has no option list, so normal chip behaviour applies."""
        data = self._data("age", MAX_UNCLEAR_ATTEMPTS - 1)
        chips = _get_chips("extraction", data, "sore_throat")
        self.assertIsNone(chips)

    def test_no_chips_outside_extraction_phase(self):
        data = self._data("paralysis_or_immobilization", MAX_UNCLEAR_ATTEMPTS - 1)
        for phase in ["routing", "context", "complete"]:
            with self.subTest(phase=phase):
                self.assertIsNone(_get_chips(phase, data, "leg_swelling"))

    def test_escalation_chips_accompany_the_rephrased_question(self):
        """The rephrased question and its options must arrive together."""
        d = LegSwellingData()
        d.age = 45
        d.active_cancer = False
        d.last_asked_field = "paralysis_or_immobilization"
        history = [{"role": "assistant",
                    "content": WELLS_QUESTIONS["paralysis_or_immobilization"]}]

        resp, d, _ = wells_extract(history, "not sure", d)
        chips = _get_chips("extraction", d.model_dump(), "leg_swelling")

        self.assertIn("a different way", resp)
        self.assertEqual(chips, WELLS_OPTIONS["paralysis_or_immobilization"])


class TestFreeTextStillAccepted(unittest.TestCase):
    """Requirement 4: ignoring the chips and typing must still work."""

    def test_typed_answer_after_chips_offered_is_honoured(self):
        d = LegSwellingData()
        d.age = 45
        d.active_cancer = False
        d.last_asked_field = "paralysis_or_immobilization"
        history = [{"role": "assistant",
                    "content": WELLS_QUESTIONS["paralysis_or_immobilization"]}]

        # Unclear answer -> rephrase + chips offered
        resp, d, _ = wells_extract(history, "not sure", d)
        history += [{"role": "user", "content": "not sure"},
                    {"role": "assistant", "content": resp}]

        # Patient ignores the chips and types their own description
        _, d, _ = wells_extract(history, "my leg has been in a cast for weeks", d)
        self.assertIs(d.paralysis_or_immobilization, True)

    def test_typed_negative_free_text_after_chips_offered(self):
        # The escalation only triggers for the field the extractor is
        # actually on, i.e. the first missing one — so fill everything
        # before pitting_edema.
        d = LegSwellingData()
        d.age = 45
        for field in ["active_cancer", "paralysis_or_immobilization",
                      "bedridden_or_surgery", "localized_tenderness",
                      "entire_leg_swollen", "calf_swelling_over_3cm"]:
            setattr(d, field, False)
        d.last_asked_field = "pitting_edema"
        history = [{"role": "assistant", "content": WELLS_QUESTIONS["pitting_edema"]}]

        resp, d, _ = wells_extract(history, "not sure", d)
        self.assertIn("a different way", resp)
        history += [{"role": "user", "content": "not sure"},
                    {"role": "assistant", "content": resp}]

        _, d, _ = wells_extract(history, "no dent at all", d)
        self.assertIs(d.pitting_edema, False)


# ---------------------------------------------------------------------------
# Table hygiene
# ---------------------------------------------------------------------------

class TestOptionTableShape(unittest.TestCase):

    def test_between_two_and_four_options_per_field(self):
        for name, table in [("wells", WELLS_OPTIONS), ("centor", CENTOR_OPTIONS),
                            ("chadsvasc", CHADSVASC_OPTIONS)]:
            for field, options in table.items():
                with self.subTest(table=name, field=field):
                    self.assertGreaterEqual(len(options), 2)
                    self.assertLessEqual(len(options), 4)

    def test_options_are_non_empty_and_unique(self):
        for name, table in [("wells", WELLS_OPTIONS), ("centor", CENTOR_OPTIONS),
                            ("chadsvasc", CHADSVASC_OPTIONS)]:
            for field, options in table.items():
                with self.subTest(table=name, field=field):
                    self.assertEqual(len(options), len(set(options)))
                    for label in options:
                        self.assertTrue(label.strip())

    def test_every_boolean_criterion_has_options(self):
        """Only the free-numeric `age` field may lack an option list."""
        for questions, options, name in [
            (WELLS_QUESTIONS, WELLS_OPTIONS, "wells"),
            (CENTOR_QUESTIONS, CENTOR_OPTIONS, "centor"),
            (CHADSVASC_QUESTIONS, CHADSVASC_OPTIONS, "chadsvasc"),
        ]:
            for field in questions:
                if field == "age":
                    continue
                with self.subTest(table=name, field=field):
                    self.assertIn(field, options)

    def test_age_deliberately_has_no_options(self):
        self.assertIsNone(get_escalation_options("sore_throat", "age"))
        self.assertIsNone(get_escalation_options("afib_stroke", "age"))

    def test_accessor_returns_a_copy(self):
        """Callers must not be able to mutate the source table."""
        got = get_escalation_options("leg_swelling", "pitting_edema")
        got.append("tampered")
        self.assertNotIn("tampered", WELLS_OPTIONS["pitting_edema"])

    def test_unknown_category_or_field_returns_none(self):
        self.assertIsNone(get_escalation_options("nope", "fever"))
        self.assertIsNone(get_escalation_options("leg_swelling", "nope"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
