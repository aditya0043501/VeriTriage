"""
Regression tests for two walkthrough-found bugs.

Bug 1 — Double "not sure" on afib_confused crashed the whole session:
  the gate asked a byte-identical question every turn (it returns early,
  before the escalation code), so main.py's top-level circuit breaker
  fired on an exact repeat, found afib_confirmed absent from
  CHADSVASC_QUESTIONS, and surfaced "Something went wrong with this
  assessment. Let's restart" — ending the session as if it were expected
  behaviour. Fix: attempt 2 gets a plainer, more direct question, and
  attempt 3 halts this assessment clearly without discarding state.

Bug 2 — "I don't know what X is" was treated as self-uncertainty:
  UNCERTAINTY_MARKERS ("don't know") caught definition requests, so the
  patient asking what a term meant just got the same question repeated.
  Fix: definition requests get a static plain-language definition, then
  the pending question is re-asked — without consuming an escalation
  attempt.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.afib_extractor import (  # noqa: E402
    extract_and_update_data as afib_extract,
    AFibStrokeData,
    AFIB_GATE_QUESTION,
    AFIB_DIRECT_QUESTION,
    AFIB_CANNOT_PROCEED_MESSAGE,
)
from extraction.rule_fallback import (  # noqa: E402
    detect_definition_request,
    build_definition_reply,
    TERM_DEFINITIONS,
    WELLS_QUESTIONS,
    CENTOR_QUESTIONS,
    CHADSVASC_QUESTIONS,
)
from extraction.leg_swelling_extractor import (  # noqa: E402
    extract_and_update_data as wells_extract,
    LegSwellingData,
)
from extraction.sore_throat_extractor import (  # noqa: E402
    extract_and_update_data as centor_extract,
    SoreThroatData,
)


def _afib_turn(data, text, history):
    resp, data, done = afib_extract(history, text, data)
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": resp},
    ]
    return resp, data, history, done


# ---------------------------------------------------------------------------
# Bug 1 — the AFib gate
# ---------------------------------------------------------------------------

class TestAfibGateNeverRepeatsVerbatim(unittest.TestCase):

    def test_double_not_sure_does_not_crash_or_restart(self):
        """The exact reproduction: two "not sure" answers in a row."""
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]

        for text in ["not sure", "not sure"]:
            resp, d, history, done = _afib_turn(d, text, history)
            # The fatal strings from the bug must never appear.
            self.assertNotIn("Something went wrong", resp)
            self.assertNotIn("Let's restart", resp)

    def test_three_unclear_answers_produce_three_distinct_messages(self):
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        seen = []
        for _ in range(3):
            resp, d, history, done = _afib_turn(d, "not sure", history)
            seen.append(resp)
        self.assertEqual(len(seen), len(set(seen)))

    def test_attempt_two_is_the_plain_direct_question(self):
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        _afib_turn(d, "not sure", history)
        resp2, d, history, done = _afib_turn(d, "not sure", history)
        self.assertIn("Let me ask it as plainly as I can", resp2)
        self.assertNotIn("Something went wrong", resp2)
        self.assertFalse(done)

    def test_attempt_three_halts_gracefully(self):
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        resp = None
        for _ in range(3):
            resp, d, history, done = _afib_turn(d, "not sure", history)
        self.assertTrue(done)
        self.assertEqual(resp, AFIB_CANNOT_PROCEED_MESSAGE)
        # It must say why, and must not masquerade as an error.
        self.assertIn("diagnosed with atrial fibrillation", resp)
        self.assertNotIn("Something went wrong", resp)
        self.assertNotIn("restart", resp.lower())

    def test_prior_state_survives_the_graceful_halt(self):
        """The bug discarded everything the patient had said. State is held
        in conv_state in main.py, so the extractor must not clear or
        corrupt its data on the halt path."""
        d = AFibStrokeData()
        d.age = 68
        d.sex = "female"
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        for _ in range(3):
            _, d, history, done = _afib_turn(d, "not sure", history)
        self.assertTrue(done)
        self.assertEqual(d.age, 68)
        self.assertEqual(d.sex, "female")
        self.assertIsNone(d.afib_confirmed)  # not guessed

    def test_afib_confirmed_is_never_defaulted(self):
        """No safe direction exists, so it must stay None — never True or False."""
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        for _ in range(5):
            _, d, history, done = _afib_turn(d, "not sure", history)
            self.assertIsNone(d.afib_confirmed)
        self.assertNotIn("afib_confirmed", d.unresolved_fields)

    def test_attempt_counter_is_actually_used(self):
        """Regression pin for the root cause: unclear_counts was {} forever
        because the gate never reached the escalation code."""
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        _afib_turn(d, "not sure", history)
        self.assertEqual(d.unclear_counts.get("afib_confirmed"), 1)

    def test_a_late_yes_still_completes_the_gate(self):
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        _afib_turn(d, "not sure", history)
        _afib_turn(d, "not sure", history)
        resp, d, history, done = _afib_turn(d, "yes", history)
        self.assertTrue(d.afib_confirmed)
        self.assertFalse(done)
        self.assertIn("Thank you for confirming", resp)


# ---------------------------------------------------------------------------
# Bug 2 — definition requests
# ---------------------------------------------------------------------------

class TestDefinitionDetection(unittest.TestCase):

    def test_misspelled_term_still_detected(self):
        """The exact reproduction, misspelling included."""
        self.assertEqual(
            detect_definition_request("i dont know what is artrial fibrillation"),
            "atrial_fibrillation",
        )

    def test_common_definition_requests(self):
        cases = {
            "what is atrial fibrillation?": "atrial_fibrillation",
            "what's afib": "atrial_fibrillation",
            "what does TIA mean": "tia",
            "I have never heard of pitting edema": "pitting_edema",
            "what is exudate": "exudate",
            "no idea what collateral veins are": "collateral_veins",
            "can you explain what a mini-stroke is": "tia",
            "what is hypertension": "hypertension",
        }
        for text, key in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_definition_request(text), key)

    def test_self_uncertainty_is_not_a_definition_request(self):
        """'I don't know IF I have X' must stay on the hedge path."""
        for text in [
            "I don't know if I have atrial fibrillation",
            "not sure",
            "I don't know",
            "maybe",
            "I don't know if I've had a stroke",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(detect_definition_request(text))

    def test_plain_answers_are_not_definition_requests(self):
        for text in ["yes", "no", "yes I have diabetes", "no fever"]:
            with self.subTest(text=text):
                self.assertIsNone(detect_definition_request(text))


class TestDefinitionReply(unittest.TestCase):

    def test_replies_with_static_definition_then_reasks(self):
        reply = build_definition_reply("pitting_edema", WELLS_QUESTIONS["pitting_edema"])
        self.assertIn(TERM_DEFINITIONS["pitting_edema"], reply)
        self.assertIn(WELLS_QUESTIONS["pitting_edema"], reply)

    def test_definitions_define_not_diagnose(self):
        """No definition may tell the patient whether they have anything."""
        forbidden = ["you have", "you may have", "you might have", "you could have"]
        for key, definition in TERM_DEFINITIONS.items():
            with self.subTest(term=key):
                for phrase in forbidden:
                    self.assertNotIn(phrase, definition.lower())

    def test_definitions_cover_terms_used_in_questions(self):
        """Every clinical term in the question tables should be definable."""
        must_define = [
            "atrial_fibrillation", "tia", "heart_failure", "hypertension",
            "vascular_disease", "diabetes", "pitting_edema",
            "collateral_veins", "bedridden", "tonsils", "exudate",
            "lymph_nodes", "strep_throat",
        ]
        for key in must_define:
            with self.subTest(term=key):
                self.assertIn(key, TERM_DEFINITIONS)
                self.assertTrue(TERM_DEFINITIONS[key].strip())


class TestDefinitionFlowInExtractors(unittest.TestCase):
    """A definition request must trigger the definition, not the
    hedge/rephrase path, in all three instruments."""

    def test_wells_definition_request(self):
        d = LegSwellingData()
        d.age = 45
        d.active_cancer = False
        d.last_asked_field = "pitting_edema"
        history = [{"role": "assistant", "content": WELLS_QUESTIONS["pitting_edema"]}]

        resp, d, done = wells_extract(history, "what is pitting edema", d)

        self.assertIn(TERM_DEFINITIONS["pitting_edema"], resp)
        self.assertFalse(done)
        # The field stays unanswered and NO escalation attempt is consumed.
        self.assertIsNone(d.pitting_edema)
        self.assertNotIn("pitting_edema", d.unclear_counts)

    def test_centor_definition_request(self):
        d = SoreThroatData()
        d.age = 30
        d.fever = True
        d.last_asked_field = "tonsillar_exudate"
        history = [{"role": "assistant", "content": CENTOR_QUESTIONS["tonsillar_exudate"]}]

        resp, d, done = centor_extract(history, "what are tonsils?", d)

        self.assertIn(TERM_DEFINITIONS["tonsils"], resp)
        self.assertIsNone(d.tonsillar_exudate)
        self.assertNotIn("tonsillar_exudate", d.unclear_counts)

    def test_afib_definition_request_at_the_gate(self):
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]

        resp, d, done = afib_extract(history, "i dont know what is artrial fibrillation", d)

        self.assertIn(TERM_DEFINITIONS["atrial_fibrillation"], resp)
        self.assertIn(AFIB_GATE_QUESTION, resp)
        self.assertFalse(done)
        # Not consumed as an unclear attempt, not treated as a hedge answer.
        self.assertNotIn("afib_confirmed", d.unclear_counts)
        self.assertIsNone(d.afib_confirmed)

    def test_definition_then_answer_continues_normally(self):
        d = AFibStrokeData()
        history = [{"role": "assistant", "content": AFIB_GATE_QUESTION}]
        _, d, history, _ = _afib_turn(d, "i dont know what is artrial fibrillation", history)
        _, d, history, _ = _afib_turn(d, "yes", history)
        self.assertTrue(d.afib_confirmed)

    def test_definition_reply_is_not_flagged_as_a_repeat(self):
        """A definition reply embeds the pending question, which previously
        made repeat-detection tools treat it as a stuck loop."""
        from extraction.extraction_utils import is_clarifying_response, is_repeat_question

        d = LegSwellingData()
        d.age = 45
        d.active_cancer = False
        d.last_asked_field = "pitting_edema"
        history = [{"role": "assistant", "content": WELLS_QUESTIONS["pitting_edema"]}]
        resp, d, done = wells_extract(history, "what is pitting edema", d)

        self.assertTrue(is_clarifying_response(resp))
        self.assertFalse(
            is_repeat_question(resp, history, is_clarifying_reprompt=True)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
