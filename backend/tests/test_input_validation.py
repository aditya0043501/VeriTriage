"""
Tests for two input-validation fixes.

Bug 1 — absurd ages accepted: every age-extraction path is bounds-checked
(10-120) after parsing. Out-of-range numbers are never stored. Centor:
invalid-age message, then unresolved after 2 attempts (age only drives the
McIsaac modifier). AFib: invalid-age message but NO give-up (age is scored
in CHA₂DS₂-VASc — proceeding without it would fabricate points).

Bug 2 — "hi" starts a pathway: a vague/greeting pre-check now runs before
keyword routing, so ultra-short non-clinical input goes to the clarifying
flow and never reaches category matching.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import classify_complaint  # noqa: E402
from extraction.rule_fallback import (  # noqa: E402
    extract_age,
    age_in_range,
    has_out_of_range_age_number,
    INVALID_AGE_MESSAGE,
)
from extraction.sore_throat_extractor import (  # noqa: E402
    extract_and_update_data as centor_extract,
    SoreThroatData,
    CENTOR_QUESTIONS,
)
from extraction.afib_extractor import (  # noqa: E402
    extract_and_update_data as afib_extract,
    AFibStrokeData,
)
from extraction.leg_swelling_extractor import (  # noqa: E402
    extract_and_update_data as wells_extract,
    LegSwellingData,
)


# ---------------------------------------------------------------------------
# Bug 1 — age bounds
# ---------------------------------------------------------------------------

class TestAgeBounds(unittest.TestCase):

    def test_shared_extract_age_rejects_out_of_range(self):
        for text, want in [("I'm 45", 45), ("21 years old", 21),
                           ("I'm 2500", None), ("999 years old", None),
                           ("I'm 5", None)]:
            with self.subTest(text=text):
                self.assertEqual(extract_age(text), want)

    def test_age_in_range_boundaries(self):
        self.assertTrue(age_in_range(10))
        self.assertTrue(age_in_range(120))
        self.assertFalse(age_in_range(9))
        self.assertFalse(age_in_range(121))
        self.assertFalse(age_in_range(None))

    def test_out_of_range_detector(self):
        self.assertTrue(has_out_of_range_age_number("im 2500"))
        self.assertTrue(has_out_of_range_age_number("I am 999 years old"))
        self.assertFalse(has_out_of_range_age_number("im 25"))
        self.assertFalse(has_out_of_range_age_number("no"))
        self.assertFalse(has_out_of_range_age_number("not sure"))

    def test_invalid_message_text(self):
        self.assertIn("between 10 and 120", INVALID_AGE_MESSAGE)


class TestCentorAgeBoundsFlow(unittest.TestCase):

    def _turn(self, d, text, history):
        resp, d, done = centor_extract(history, text, d)
        history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": resp},
        ]
        return resp, d, history, done

    def _awaiting_age(self):
        d = SoreThroatData()
        d.last_asked_field = "age"
        history = [{"role": "assistant", "content": CENTOR_QUESTIONS["age"]}]
        return d, history

    def test_absurd_age_not_stored_and_invalid_message(self):
        d, history = self._awaiting_age()
        resp, d, history, _ = self._turn(d, "im 2500", history)
        self.assertIsNone(d.age)
        self.assertIn("That doesn't look like a valid age", resp)

    def test_valid_age_accepted(self):
        d, history = self._awaiting_age()
        resp, d, history, _ = self._turn(d, "im 25", history)
        self.assertEqual(d.age, 25)
        self.assertNotIn("valid age", resp)

    def test_two_out_of_range_attempts_mark_unresolved(self):
        d, history = self._awaiting_age()
        self._turn(d, "im 2500", history)
        resp, d, history, _ = self._turn(d, "3000", history)
        self.assertIn("age", d.unresolved_fields)
        self.assertIsNone(d.age)
        self.assertIn("continue without it", resp)

    def test_recovery_after_invalid_age(self):
        """A valid age after one invalid answer is accepted normally."""
        d, history = self._awaiting_age()
        self._turn(d, "im 2500", history)
        _, d, history, _ = self._turn(d, "im 25", history)
        self.assertEqual(d.age, 25)
        self.assertNotIn("age", d.unresolved_fields)


class TestOtherPathwaysAgeBounds(unittest.TestCase):

    def test_wells_does_not_store_absurd_age(self):
        d = LegSwellingData()
        d.last_asked_field = "age"
        history = [{"role": "assistant", "content": "How old are you?"}]
        _, d, _ = wells_extract(history, "I'm 2500 years old", d)
        self.assertIsNone(d.age)

    def test_afib_does_not_store_absurd_age_and_warns(self):
        """AFib: no give-up — age is a scored CHA₂DS₂-VASc criterion."""
        d = AFibStrokeData()
        d.afib_confirmed = True
        d.last_asked_field = "age"
        history = [{"role": "assistant", "content": "How old are you?"}]
        resp, d, _ = afib_extract(history, "I'm 2500 years old", d)
        self.assertIsNone(d.age)
        self.assertIn("That doesn't look like a valid age", resp)
        self.assertNotIn("age", d.unresolved_fields)

    def test_afib_valid_age_accepted(self):
        d = AFibStrokeData()
        d.afib_confirmed = True
        d.last_asked_field = "age"
        history = [{"role": "assistant", "content": "How old are you?"}]
        _, d, _ = afib_extract(history, "I'm 72", d)
        self.assertEqual(d.age, 72)


# ---------------------------------------------------------------------------
# Bug 2 — greeting/short-input pre-check
# ---------------------------------------------------------------------------

class TestVaguePrecheck(unittest.TestCase):

    def test_greetings_and_acks_route_to_vague(self):
        for text in ["hi", "hello", "hey", "ok", "thanks", "hello there"]:
            with self.subTest(text=text):
                self.assertEqual(classify_complaint(text), "vague")

    def test_keyword_in_short_input_still_routes(self):
        """The pre-check must not swallow real clinical keywords."""
        self.assertEqual(classify_complaint("hello my leg hurts"), "leg_swelling")
        self.assertEqual(classify_complaint("sore throat"), "sore_throat")
        self.assertEqual(classify_complaint("bad cough"), "pneumonia")

    def test_out_of_scope_short_input_still_out_of_scope(self):
        """A named out-of-scope symptom keeps its dedicated message, not vague."""
        self.assertEqual(classify_complaint("headache"), "out_of_scope")

    def test_vague_then_specific_resumes_normal_routing(self):
        """'hi' then 'leg pain' must route normally on the second turn."""
        self.assertEqual(classify_complaint("hi"), "vague")
        self.assertEqual(classify_complaint("hi leg pain"), "leg_swelling")

    def test_multword_non_clinical_still_vague(self):
        self.assertEqual(classify_complaint("I'm not feeling well at all"), "vague")


if __name__ == "__main__":
    unittest.main(verbosity=2)
