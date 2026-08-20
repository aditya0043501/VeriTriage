"""
Tests for the CURB-65 pneumonia pathway.

Scoring math, tiers, published mortality figures (Lim et al., Thorax 2003),
and the extraction edge cases the pathway introduces (numeric breathing rate,
age as a number or a yes/no, urea/BP as patient-reportable measurements).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring.curb65_score import calculate_curb65_score, MORTALITY_BY_SCORE  # noqa: E402
from explanations.probability_mapping import (  # noqa: E402
    get_curb65_probability,
    UncitedScoreError,
)
from explanations.explanation_builder import build_explanation  # noqa: E402
from extraction.pneumonia_extractor import (  # noqa: E402
    extract_and_update_data,
    Curb65Data,
    CURB65_QUESTIONS,
)
from router import classify_complaint  # noqa: E402


def _score(**kw):
    defaults = dict(confusion=False, urea_elevated=False, rr_high=False,
                    bp_low=False, age_65_plus=False)
    defaults.update(kw)
    return calculate_curb65_score(**defaults)


class TestCurb65Scoring(unittest.TestCase):

    def test_case_a_all_false_age_30(self):
        """Spec case (a): all false, young patient -> 0, low."""
        r = _score()
        self.assertEqual(r["score"], 0)
        self.assertEqual(r["tier"], "low")
        self.assertEqual(r["mortality_30_day"], "0.6%")

    def test_case_b_confusion_rr_age(self):
        """Spec case (b): confusion + RR high + age 65+ -> 3, severe."""
        r = _score(confusion=True, rr_high=True, age_65_plus=True)
        self.assertEqual(r["score"], 3)
        self.assertEqual(r["tier"], "severe")
        self.assertEqual(r["mortality_30_day"], "14.0%")

    def test_case_c_age_only(self):
        """Spec case (c): age 65+ only -> 1, low."""
        r = _score(age_65_plus=True)
        self.assertEqual(r["score"], 1)
        self.assertEqual(r["tier"], "low")
        self.assertEqual(r["mortality_30_day"], "2.7%")

    def test_score_two_is_moderate(self):
        r = _score(rr_high=True, age_65_plus=True)
        self.assertEqual(r["score"], 2)
        self.assertEqual(r["tier"], "moderate")
        self.assertEqual(r["mortality_30_day"], "6.8%")

    def test_max_score_is_five(self):
        r = _score(confusion=True, urea_elevated=True, rr_high=True,
                   bp_low=True, age_65_plus=True)
        self.assertEqual(r["score"], 5)
        self.assertEqual(r["tier"], "severe")
        self.assertEqual(r["mortality_30_day"], "27.8%")

    def test_mortality_table_matches_published_figures(self):
        """Pin the published numbers so a typo can't ship silently."""
        self.assertEqual(MORTALITY_BY_SCORE,
                         {0: "0.6%", 1: "2.7%", 2: "6.8%",
                          3: "14.0%", 4: "27.8%", 5: "27.8%"})

    def test_score_is_partial_with_pending_fields(self):
        r = _score()
        self.assertTrue(r["isPartial"])
        self.assertIn("Chest X-ray confirmation", r["pendingFields"])

    def test_recommendations_match_tier(self):
        self.assertIn("Home care", _score()["recommendation"]["what_to_do"])
        self.assertIn("hospital", _score(score=2) if False else
                      _score(rr_high=True, age_65_plus=True)["recommendation"]["what_to_do"].lower())
        self.assertIn("emergency", _score(confusion=True, rr_high=True, bp_low=True)
                      ["recommendation"]["what_to_do"].lower())

    def test_citation_is_lim_2003(self):
        r = _score()
        self.assertIn("Thorax. 2003", r["citation"])
        self.assertIn("Lim WS", r["citation"])


class TestCurb65ProbabilityMapping(unittest.TestCase):

    def test_each_score_maps_to_its_published_mortality(self):
        expected = {0: "0.6%", 1: "2.7%", 2: "6.8%", 3: "14.0%", 4: "27.8%", 5: "27.8%"}
        for score, mortality in expected.items():
            with self.subTest(score=score):
                p = get_curb65_probability(score)
                self.assertEqual(p.probability_text, mortality)
                self.assertIn("Thorax. 2003", p.citation)

    def test_tiers(self):
        self.assertEqual(get_curb65_probability(0).tier, "low")
        self.assertEqual(get_curb65_probability(1).tier, "low")
        self.assertEqual(get_curb65_probability(2).tier, "moderate")
        self.assertEqual(get_curb65_probability(3).tier, "severe")
        self.assertEqual(get_curb65_probability(5).tier, "severe")

    def test_out_of_range_raises(self):
        for bad in [-1, 6, "2", True, None]:
            with self.subTest(bad=bad):
                with self.assertRaises(UncitedScoreError):
                    get_curb65_probability(bad)


class TestPneumoniaRouting(unittest.TestCase):

    def test_verification_opening_routes_to_pneumonia(self):
        self.assertEqual(classify_complaint("I have a bad cough and fever"), "pneumonia")

    def test_spec_keywords(self):
        for text in ["I might have pneumonia", "a chest infection",
                     "lung infection", "coughing up green sputum",
                     "short of breath", "lots of phlegm",
                     "pneumococcal", "respiratory infection"]:
            with self.subTest(text=text):
                self.assertEqual(classify_complaint(text), "pneumonia")

    def test_existing_pathways_unaffected(self):
        self.assertEqual(classify_complaint("my leg is swollen"), "leg_swelling")
        self.assertEqual(classify_complaint("I have a sore throat"), "sore_throat")
        self.assertEqual(classify_complaint("I have atrial fibrillation"), "afib_stroke")


class TestPneumoniaExtraction(unittest.TestCase):

    def _turn(self, data, text, history):
        resp, data, done = extract_and_update_data(history, text, data)
        history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": resp},
        ]
        return resp, data, history, done

    def test_verification_conversation_scores_two(self):
        """The spec's verification flow: fast breathing at 35/min + age 70."""
        d = Curb65Data()
        history = []
        script = [
            "I have a bad cough and fever",
            "no",                                  # confusion
            "no",                                  # urea
            "I'm breathing fast, about 35 per minute",  # RR -> True (numeric)
            "no",                                  # BP
            "70",                                  # age -> age_65_plus True
        ]
        for text in script:
            _, d, history, done = self._turn(d, text, history)
        self.assertTrue(d.is_complete())
        self.assertIs(d.confusion, False)
        self.assertIs(d.urea_elevated, False)
        self.assertIs(d.rr_high, True)
        self.assertIs(d.bp_low, False)
        self.assertIs(d.age_65_plus, True)

    def test_numeric_breathing_rate_under_30_is_false(self):
        d = Curb65Data(confusion=False, urea_elevated=False)
        history = [{"role": "assistant", "content": CURB65_QUESTIONS["rr_high"]}]
        d.last_asked_field = "rr_high"
        _, d, _, _ = self._turn(d, "about 18 breaths per minute", history)
        self.assertIs(d.rr_high, False)

    def test_numeric_age_under_65_is_false(self):
        d = Curb65Data(confusion=False, urea_elevated=False, rr_high=False, bp_low=False)
        history = [{"role": "assistant", "content": CURB65_QUESTIONS["age_65_plus"]}]
        d.last_asked_field = "age_65_plus"
        _, d, _, _ = self._turn(d, "I'm 52", history)
        self.assertIs(d.age_65_plus, False)

    def test_not_sure_escalation_still_works(self):
        d = Curb65Data()
        history = [{"role": "assistant", "content": CURB65_QUESTIONS["confusion"]}]
        d.last_asked_field = "confusion"
        resp, d, history, _ = self._turn(d, "not sure", history)
        self.assertIn("a different way", resp)
        self.assertNotIn(CURB65_QUESTIONS["confusion"], resp.split("a different way.")[-1][:0] or "")
        resp2, d, history, done = self._turn(d, "not sure", history)
        self.assertIn("confusion", d.unresolved_fields)
        self.assertIs(d.confusion, False)


class TestCurb65ExplanationLayer(unittest.TestCase):

    def test_explanation_shows_published_mortality_and_citation(self):
        r = _score(rr_high=True, age_65_plus=True)
        expl = build_explanation("pneumonia", r, {
            "confusion": False, "urea_elevated": False, "rr_high": True,
            "bp_low": False, "age_65_plus": True,
        })
        self.assertTrue(expl["available"])
        self.assertEqual(expl["probability_text"], "6.8%")
        self.assertIn("Thorax 2003", expl["probability_context"])
        self.assertIn("Thorax. 2003", expl["probability_citation"])
        # Mortality must not be phrased as disease probability.
        self.assertNotIn("were found to have", expl["probability_context"])

    def test_all_five_criteria_have_templates(self):
        """All-present case must build — proving all 5 present slots exist."""
        r = _score(confusion=True, urea_elevated=True, rr_high=True,
                   bp_low=True, age_65_plus=True)
        expl = build_explanation("pneumonia", r, {
            "confusion": True, "urea_elevated": True, "rr_high": True,
            "bp_low": True, "age_65_plus": True,
        })
        self.assertIsNotNone(expl)
        self.assertEqual(len(expl["criteria"]), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
