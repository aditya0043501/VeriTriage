"""
Tests for source-quote capture.

When a criterion matches positively, the patient's triggering words are
recorded verbatim and flow through to the explanation layer. This is
record-keeping only: it must never change what gets extracted or scored.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.rule_fallback import find_source_quote, WELLS_PATTERNS  # noqa: E402
from extraction.leg_swelling_extractor import (  # noqa: E402
    extract_and_update_data as wells_extract,
    LegSwellingData,
    WELLS_QUESTIONS,
)
from explanations.explanation_builder import build_explanation  # noqa: E402
from scoring.wells_score import calculate_wells_score  # noqa: E402


def _wells_turn(data, text, history):
    resp, data, done = wells_extract(history, text, data)
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": resp},
    ]
    return resp, data, history, done


class TestFindSourceQuote(unittest.TestCase):

    def test_explicit_answer_to_pending_question_is_quoted(self):
        got = find_source_quote(
            ["yes it is tender"], "yes it is tender",
            "localized_tenderness", WELLS_PATTERNS["localized_tenderness"],
        )
        self.assertEqual(got, "yes it is tender")

    def test_keyword_hit_in_current_message_is_quoted(self):
        text = "it's really sore when I press on my calf"
        got = find_source_quote([text], text, "localized_tenderness",
                                WELLS_PATTERNS["localized_tenderness"])
        self.assertEqual(got, text)

    def test_earlier_turn_is_found_when_current_input_is_unrelated(self):
        turns = ["my calf looks bigger than the other one", "I'm 45"]
        got = find_source_quote(turns, "I'm 45", "calf_swelling_over_3cm",
                                WELLS_PATTERNS["calf_swelling_over_3cm"])
        self.assertEqual(got, "my calf looks bigger than the other one")

    def test_falls_back_to_current_input(self):
        got = find_source_quote(["yeah"], "yeah", "collateral_veins",
                                WELLS_PATTERNS["collateral_veins"])
        self.assertEqual(got, "yeah")


class TestExtractorCapturesQuotes(unittest.TestCase):

    def _primed(self):
        d = LegSwellingData()
        d.age = 45
        d.active_cancer = False
        d.last_asked_field = "localized_tenderness"
        history = [{"role": "assistant", "content": WELLS_QUESTIONS["localized_tenderness"]}]
        return d, history

    def test_positive_match_stores_the_triggering_message(self):
        d, history = self._primed()
        text = "it's really sore when I press on my calf"
        _, d, _, _ = _wells_turn(d, text, history)
        self.assertIs(d.localized_tenderness, True)
        self.assertEqual(d.source_quotes.get("localized_tenderness"), text)

    def test_bare_yes_answer_is_quoted_verbatim(self):
        d, history = self._primed()
        _, d, _, _ = _wells_turn(d, "yes", history)
        self.assertIs(d.localized_tenderness, True)
        self.assertEqual(d.source_quotes.get("localized_tenderness"), "yes")

    def test_negative_matches_get_no_quote(self):
        d, history = self._primed()
        _, d, _, _ = _wells_turn(d, "no, not at all", history)
        self.assertIs(d.localized_tenderness, False)
        self.assertNotIn("localized_tenderness", d.source_quotes)

    def test_first_quote_wins(self):
        """A field already quoted is not overwritten by later turns."""
        d, history = self._primed()
        text = "it's really sore when I press on my calf"
        _, d, history, _ = _wells_turn(d, text, history)
        _, d, history, _ = _wells_turn(d, "still tender", history)
        self.assertEqual(d.source_quotes.get("localized_tenderness"), text)

    def test_extraction_result_is_unchanged_by_capture(self):
        """The sidecar must not alter what is extracted."""
        d, history = self._primed()
        _, d, _, _ = _wells_turn(d, "yes, tender", history)
        self.assertIs(d.localized_tenderness, True)
        self.assertIs(d.active_cancer, False)


class TestExplanationUsesQuotes(unittest.TestCase):

    def _score_and_build(self, quotes):
        criteria = dict(
            active_cancer=False, paralysis_or_immobilization=False,
            bedridden_or_surgery=False, localized_tenderness=True,
            entire_leg_swollen=False, calf_swelling_over_3cm=True,
            pitting_edema=False, collateral_veins=False,
        )
        result = calculate_wells_score(**criteria)
        return build_explanation("leg_swelling", result, criteria, source_quotes=quotes)

    def test_real_quote_appears_in_explanation(self):
        quotes = {
            "localized_tenderness": "it's really sore when I press on my calf",
            "calf_swelling_over_3cm": "my left calf looks noticeably bigger than the right",
        }
        expl = self._score_and_build(quotes)
        words = {c["label"]: c["patient_words"] for c in expl["criteria"]}
        self.assertEqual(words["Localized tenderness"],
                         "it's really sore when I press on my calf")
        self.assertEqual(words["Calf swelling >3 cm"],
                         "my left calf looks noticeably bigger than the right")

    def test_long_quotes_are_truncated_to_80_chars(self):
        long_quote = "my calf has been really sore for days now, especially when I press on it, and it seems to be getting worse"
        self.assertGreater(len(long_quote), 80)
        expl = self._score_and_build({"localized_tenderness": long_quote})
        words = {c["label"]: c["patient_words"] for c in expl["criteria"]}
        got = words["Localized tenderness"]
        self.assertLessEqual(len(got), 83)  # 80 chars + "..."
        self.assertTrue(got.endswith("..."))

    def test_missing_quote_falls_back_to_none(self):
        """No quote -> patient_words is None; the frontend then shows
        'Based on your description'. The old placeholder never appears."""
        expl = self._score_and_build({})
        for c in expl["criteria"]:
            self.assertIsNone(c["patient_words"])
            self.assertNotIn("from your conversation", c["svg"])

    def test_quotes_appear_inside_svg_diagram(self):
        quotes = {"localized_tenderness": "sore when I press it"}
        expl = self._score_and_build(quotes)
        svg = next(c["svg"] for c in expl["criteria"]
                   if c["label"] == "Localized tenderness")
        self.assertIn("sore when I press it", svg)

    def test_unresolved_fields_carry_no_quote(self):
        """A field given up on is stored as no-contribution with no quote."""
        d = LegSwellingData()
        d.age = 45
        d.active_cancer = False
        d.last_asked_field = "paralysis_or_immobilization"
        history = [{"role": "assistant", "content": WELLS_QUESTIONS["paralysis_or_immobilization"]}]
        _, d, history, _ = _wells_turn(d, "not sure", history)
        _, d, history, _ = _wells_turn(d, "not sure", history)
        self.assertIs(d.paralysis_or_immobilization, False)
        self.assertNotIn("paralysis_or_immobilization", d.source_quotes)
        self.assertIn("paralysis_or_immobilization", d.unresolved_fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
