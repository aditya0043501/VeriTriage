"""
Regression tests for the Centor age-question loop.

Bug: "im 21" / "21" failed age parsing, the conversation jumped to other
criteria, and age was never cleanly re-asked — so is_complete() could never
become true and the pathway looped.

Fix: lax age parsing (bare 10-99) only while age is the pending question,
a text-only numeric re-ask, and an unresolved-age give-up that scores
Centor without the McIsaac modifier.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.sore_throat_extractor import (  # noqa: E402
    extract_and_update_data,
    SoreThroatData,
    CENTOR_QUESTIONS,
)
from main import _get_chips, _score_sore_throat  # noqa: E402


def _turn(data, text, history):
    resp, data, done = extract_and_update_data(history, text, data)
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": resp},
    ]
    return resp, data, history, done


def _awaiting_age():
    d = SoreThroatData()
    d.last_asked_field = "age"
    history = [{"role": "assistant", "content": CENTOR_QUESTIONS["age"]}]
    return d, history


class TestAgeParsingWhilePending(unittest.TestCase):

    def test_im_21_parses(self):
        d, history = _awaiting_age()
        _, d, _, _ = _turn(d, "im 21", history)
        self.assertEqual(d.age, 21)

    def test_bare_number_parses(self):
        d, history = _awaiting_age()
        _, d, _, _ = _turn(d, "21", history)
        self.assertEqual(d.age, 21)

    def test_age_21_parses(self):
        d, history = _awaiting_age()
        _, d, _, _ = _turn(d, "age 21", history)
        self.assertEqual(d.age, 21)

    def test_strict_forms_still_parse(self):
        d, history = _awaiting_age()
        _, d, _, _ = _turn(d, "I'm 45 years old", history)
        self.assertEqual(d.age, 45)

    def test_lax_parsing_is_gated_on_the_age_question(self):
        """A temperature must not be misread as an age when we didn't ask."""
        d = SoreThroatData()
        d.last_asked_field = "fever"
        history = [{"role": "assistant", "content": CENTOR_QUESTIONS["fever"]}]
        _, d, _, _ = _turn(d, "yes, 38.5 degrees", history)
        self.assertIsNone(d.age)

    def test_under_10_not_accepted(self):
        """The regex only matches 10-99, so a stray digit can't become an age."""
        d, history = _awaiting_age()
        _, d, _, _ = _turn(d, "7", history)
        self.assertIsNone(d.age)


class TestAgeEscalation(unittest.TestCase):

    def test_first_failure_reasks_text_only_with_spec_prompt(self):
        d, history = _awaiting_age()
        resp, d, history, _ = _turn(d, "not sure", history)
        self.assertIn("What is your age? Please enter just the number", resp)
        # No chips for a numeric question
        chips = _get_chips("extraction", d.model_dump(), "sore_throat")
        self.assertIsNone(chips)

    def test_second_failure_marks_age_unresolved_and_proceeds(self):
        d, history = _awaiting_age()
        _, d, history, _ = _turn(d, "not sure", history)
        resp, d, history, _ = _turn(d, "dunno", history)
        self.assertIn("age", d.unresolved_fields)
        self.assertIsNone(d.age)  # never defaulted to False — it's numeric
        self.assertIn("continue without it", resp)

    def test_unresolved_age_does_not_block_completion(self):
        d, history = _awaiting_age()
        _turn(d, "not sure", history)
        _, d, history, _ = _turn(d, "dunno", history)
        # Answer the four criteria
        for answer in ["yes", "yes", "no", "yes"]:
            _, d, history, done = _turn(d, answer, history)
        self.assertTrue(d.is_complete())
        self.assertTrue(done)

    def test_never_asks_age_three_times(self):
        d, history = _awaiting_age()
        responses = []
        for _ in range(4):
            resp, d, history, done = _turn(d, "not sure", history)
            responses.append(resp)
            if done:
                break
        age_asks = sum(1 for r in responses if "old are you" in r or "What is your age" in r)
        self.assertLessEqual(age_asks, 2)


class TestUnresolvedAgeScoring(unittest.TestCase):

    def test_scores_without_age_modifier(self):
        """Unresolved age scores with the neutral 15-44 band (0 modifier)."""
        class _Data:
            fever = True
            absence_of_cough = True
            tender_cervical_nodes = False
            tonsillar_exudate = False
            age = None
        r = _score_sore_throat(_Data())
        self.assertEqual(r["score"], 2)  # fever +1, no-cough +1, no age modifier
        self.assertEqual(r["breakdown"]["Age"]["points"], 0)

    def test_known_age_still_applies_modifier(self):
        class _Data:
            fever = True
            absence_of_cough = False
            tender_cervical_nodes = False
            tonsillar_exudate = False
            age = 50
        r = _score_sore_throat(_Data())
        self.assertEqual(r["score"], 0)  # fever +1, age 50 -> -1


class TestBugReportConversation(unittest.TestCase):
    """The exact 7-turn conversation from the bug report."""

    def test_full_flow_scores_after_turn_7(self):
        d = SoreThroatData()
        history = []
        script = [
            "i have a sore throat",
            "it is itchy and sore like voice has changed",
            "im 21",
            "yes i do have a fever",
            "yes i do have cough along with my fever",
            "it is not swollen near my neck but it hurts behind my ears",
            "yes i see swelling on my tonsils but i dont see any white patches",
        ]
        age_asked_after_turn3 = False
        done = False
        for i, text in enumerate(script):
            resp, d, history, done = _turn(d, text, history)
            if i >= 2 and ("old are you" in resp or "What is your age" in resp):
                age_asked_after_turn3 = True

        self.assertFalse(age_asked_after_turn3, "age was re-asked after turn 3")
        self.assertEqual(d.age, 21)
        self.assertTrue(done, "conversation should be complete after turn 7")
        # fever +1, cough present +0, nodes +0, tonsil swelling +1, age 21 -> +0
        self.assertEqual(d.fever, True)
        self.assertEqual(d.absence_of_cough, False)
        self.assertEqual(d.tender_cervical_nodes, False)
        self.assertEqual(d.tonsillar_exudate, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
