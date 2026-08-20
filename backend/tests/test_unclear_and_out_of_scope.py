"""
Tests for two behaviours added together:

PART 1 — "not sure" clarification loop
  A patient answering "not sure" must never be re-asked the identical
  sentence. The system rephrases once (static, pre-written text), then
  stops asking and records the field as unresolved.

PART 2 — out-of-scope symptom flagging
  A symptom that matches no criterion in the active instrument must be
  recorded verbatim and surfaced in the report, WITHOUT being classified
  and WITHOUT affecting extraction of the in-scope criteria.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.leg_swelling_extractor import (  # noqa: E402
    extract_and_update_data as wells_extract,
    LegSwellingData,
)
from extraction.rule_fallback import (  # noqa: E402
    WELLS_QUESTIONS,
    WELLS_REPHRASINGS,
    CENTOR_QUESTIONS,
    CENTOR_REPHRASINGS,
    CHADSVASC_QUESTIONS,
    CHADSVASC_REPHRASINGS,
    detect_out_of_scope_mentions,
    format_out_of_scope_notes,
    extract_wells_fields,
)
from extraction.extraction_utils import MAX_UNCLEAR_ATTEMPTS  # noqa: E402


def _turn(data, text, history):
    """Run one extraction turn and return (response, data, history)."""
    resp, data, done = wells_extract(history, text, data)
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": resp},
    ]
    return resp, data, history, done


def _primed_wells():
    """A Wells conversation sitting on the paralysis question."""
    d = LegSwellingData()
    d.age = 45
    d.active_cancer = False
    d.last_asked_field = "paralysis_or_immobilization"
    history = [{
        "role": "assistant",
        "content": WELLS_QUESTIONS["paralysis_or_immobilization"],
    }]
    return d, history


# ---------------------------------------------------------------------------
# PART 1 — "not sure" clarification loop
# ---------------------------------------------------------------------------

class TestNotSureRephrasing(unittest.TestCase):

    def test_first_not_sure_does_not_repeat_the_question(self):
        """The bug: 'not sure' used to return the identical sentence."""
        d, history = _primed_wells()
        original = WELLS_QUESTIONS["paralysis_or_immobilization"]

        resp, d, history, _ = _turn(d, "not sure", history)

        self.assertNotEqual(resp.strip(), original.strip())
        self.assertNotIn(original, resp)

    def test_first_not_sure_uses_the_static_rephrasing(self):
        d, history = _primed_wells()
        resp, d, history, _ = _turn(d, "not sure", history)
        self.assertIn(WELLS_REPHRASINGS["paralysis_or_immobilization"], resp)

    def test_second_not_sure_stops_asking_and_moves_on(self):
        d, history = _primed_wells()
        resp1, d, history, _ = _turn(d, "not sure", history)
        resp2, d, history, _ = _turn(d, "not sure", history)

        # Moved on to the next criterion rather than asking a third time.
        self.assertNotIn(WELLS_REPHRASINGS["paralysis_or_immobilization"], resp2)
        self.assertIn(WELLS_QUESTIONS["bedridden_or_surgery"], resp2)
        self.assertEqual(d.last_asked_field, "bedridden_or_surgery")

    def test_unresolved_field_is_recorded_not_silently_answered_no(self):
        d, history = _primed_wells()
        _, d, history, _ = _turn(d, "not sure", history)
        _, d, history, _ = _turn(d, "not sure", history)

        self.assertIn("paralysis_or_immobilization", d.unresolved_fields)
        # Scored as no-contribution (conservative), but flagged as unresolved.
        self.assertIs(d.paralysis_or_immobilization, False)

    def test_never_loops_the_same_question_across_many_not_sures(self):
        """Six consecutive 'not sure' answers must never repeat a question."""
        d, history = _primed_wells()
        seen = []
        for _ in range(6):
            resp, d, history, done = _turn(d, "not sure", history)
            seen.append(resp.strip())
            if done:
                break
        self.assertEqual(len(seen), len(set(seen)), f"repeated prompt in: {seen}")

    def test_attempt_budget_is_two(self):
        """Guard the documented escalation budget."""
        self.assertEqual(MAX_UNCLEAR_ATTEMPTS, 2)

    def test_a_clear_answer_after_not_sure_is_still_honoured(self):
        """Rephrasing must not break normal extraction."""
        d, history = _primed_wells()
        _, d, history, _ = _turn(d, "not sure", history)
        _, d, history, _ = _turn(d, "yes, my leg is in a cast", history)

        self.assertIs(d.paralysis_or_immobilization, True)
        self.assertNotIn("paralysis_or_immobilization", d.unresolved_fields)


class TestRephrasingTablesAreComplete(unittest.TestCase):
    """Every question must have a pre-written rephrasing, and it must
    actually differ from the original."""

    def _check(self, questions, rephrasings, name):
        for field, original in questions.items():
            with self.subTest(table=name, field=field):
                self.assertIn(field, rephrasings)
                self.assertTrue(rephrasings[field].strip())
                self.assertNotEqual(
                    rephrasings[field].strip().lower(),
                    original.strip().lower(),
                )

    def test_wells(self):
        self._check(WELLS_QUESTIONS, WELLS_REPHRASINGS, "wells")

    def test_centor(self):
        self._check(CENTOR_QUESTIONS, CENTOR_REPHRASINGS, "centor")

    def test_chadsvasc(self):
        self._check(CHADSVASC_QUESTIONS, CHADSVASC_REPHRASINGS, "chadsvasc")


# ---------------------------------------------------------------------------
# PART 2 — out-of-scope symptom flagging
# ---------------------------------------------------------------------------

class TestOutOfScopeDetection(unittest.TestCase):

    def test_captures_unrelated_symptom_alongside_in_scope_one(self):
        text = "my calf is swollen and I also have some numbness in my foot"
        found = detect_out_of_scope_mentions(text, "leg_swelling")
        self.assertEqual(len(found), 1)
        self.assertIn("numbness in my foot", found[0])

    def test_captures_verbatim_patient_words(self):
        """The mention must be the patient's own words, not a rewrite."""
        text = "the leg is swollen. I have been getting headaches too"
        found = detect_out_of_scope_mentions(text, "leg_swelling")
        self.assertEqual(found, ["I have been getting headaches too"])

    def test_does_not_flag_in_scope_criteria(self):
        for text in [
            "My left calf is really sore when I press on it",
            "my whole leg is swollen",
            "yes, it leaves a dent",
            "I have new veins on the surface of my leg",
        ]:
            with self.subTest(text=text):
                self.assertEqual(detect_out_of_scope_mentions(text, "leg_swelling"), [])

    def test_does_not_flag_plain_answers(self):
        for text in ["yes", "no", "not sure", "no, it does not leave a dent"]:
            with self.subTest(text=text):
                self.assertEqual(detect_out_of_scope_mentions(text, "leg_swelling"), [])

    def test_does_not_flag_denied_symptoms(self):
        text = "I don't have any numbness in my foot"
        self.assertEqual(detect_out_of_scope_mentions(text, "leg_swelling"), [])

    def test_works_for_centor(self):
        text = "my throat is killing me and I have a rash on my chest"
        found = detect_out_of_scope_mentions(text, "sore_throat")
        self.assertEqual(found, ["I have a rash on my chest"])

    def test_works_for_chadsvasc(self):
        text = "I have afib. I have also noticed some blurry vision lately"
        found = detect_out_of_scope_mentions(text, "afib_stroke")
        self.assertEqual(len(found), 1)
        self.assertIn("blurry vision", found[0])

    def test_unknown_category_returns_nothing(self):
        self.assertEqual(detect_out_of_scope_mentions("I have a headache", "nope"), [])

    def test_duplicate_mentions_are_deduped(self):
        text = "I have a rash on my chest. I have a rash on my chest"
        found = detect_out_of_scope_mentions(text, "sore_throat")
        self.assertEqual(len(found), 1)


class TestOutOfScopeDoesNotAffectScoring(unittest.TestCase):
    """The whole point: flagging must be a side-channel, never a change
    to what the rule engine extracts."""

    def test_wells_extraction_identical_with_and_without_extra_mention(self):
        clean = "my calf is really tender when I press on it"
        noisy = "my calf is really tender when I press on it and I also have some numbness in my foot"
        missing = ["localized_tenderness"]

        got_clean, _ = extract_wells_fields(clean, clean, missing, "localized_tenderness")
        got_noisy, _ = extract_wells_fields(noisy, noisy, missing, "localized_tenderness")

        self.assertEqual(got_clean, got_noisy)
        self.assertIs(got_noisy.get("localized_tenderness"), True)

    def test_full_turn_scores_in_scope_part_and_flags_the_rest(self):
        d = LegSwellingData()
        d.age = 45
        d.last_asked_field = "localized_tenderness"
        history = [{"role": "assistant", "content": WELLS_QUESTIONS["localized_tenderness"]}]

        text = "yes it is tender, and I also have some numbness in my foot"
        _, d, _, _ = _turn(d, text, history)

        self.assertIs(d.localized_tenderness, True)
        self.assertEqual(
            len(detect_out_of_scope_mentions(text, "leg_swelling")), 1
        )


class TestOutOfScopeNoteWording(unittest.TestCase):
    """The note must state only that we didn't screen for it."""

    FORBIDDEN = [
        "could be", "may be", "might be", "suggests", "consistent with",
        "indicates", "diagnos", "likely", "probably", "sign of",
        "neuropathy", "sciatica", "infection",
    ]

    def test_no_mentions_returns_none(self):
        self.assertIsNone(format_out_of_scope_notes([], "leg_swelling"))

    def test_note_quotes_patient_words_verbatim(self):
        note = format_out_of_scope_notes(["I have some numbness in my foot"], "leg_swelling")
        self.assertIn('"I have some numbness in my foot"', note)

    def test_note_says_we_did_not_screen_for_it(self):
        note = format_out_of_scope_notes(["I have a headache"], "leg_swelling")
        self.assertIn("didn't screen for", note)
        self.assertIn("worth mentioning to your doctor", note)

    def test_note_never_speculates_about_a_condition(self):
        for category in ["leg_swelling", "sore_throat", "afib_stroke"]:
            note = format_out_of_scope_notes(
                ["I have numbness in my foot", "I have been getting headaches"],
                category,
            )
            low = note.lower()
            for phrase in self.FORBIDDEN:
                with self.subTest(category=category, phrase=phrase):
                    self.assertNotIn(phrase, low)

    def test_note_lists_multiple_mentions(self):
        note = format_out_of_scope_notes(
            ["I have a headache", "I have been feeling dizzy"], "leg_swelling"
        )
        self.assertIn('"I have a headache"', note)
        self.assertIn('"I have been feeling dizzy"', note)


if __name__ == "__main__":
    unittest.main(verbosity=2)
