"""
Tests for the emergency triage gate.

The gate runs as the FIRST line of /api/chat processing. A trigger returns
the fixed emergency message and halts everything — no routing, no scoring,
no conversation-state mutation.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio  # noqa: E402

from emergency_triage import get_emergency_message, EMERGENCY_MESSAGE  # noqa: E402
from main import chat, conversations, ChatRequest  # noqa: E402


def _post(message: str, conversation_id: str):
    """Call the chat endpoint directly (TestClient is unusable with the
    pinned httpx/starlette versions; this exercises the same code path)."""
    return asyncio.run(chat(ChatRequest(message=message, conversation_id=conversation_id)))


class TestTriggerTable(unittest.TestCase):
    """Every approved trigger group fires; exact fixed message returned."""

    def test_group_1_chest_pain_plus_breathing(self):
        self.assertEqual(get_emergency_message("I have chest pain and can't breathe"),
                         EMERGENCY_MESSAGE)
        self.assertEqual(get_emergency_message("chest pain and I'm short of breath"),
                         EMERGENCY_MESSAGE)

    def test_group_1_requires_both_parts(self):
        """Chest pain alone is out-of-scope for the tool, but not an
        emergency trigger on its own per the approved list."""
        self.assertIsNone(get_emergency_message("I have chest pain"))

    def test_group_2_cardiac(self):
        for text in ["I think I'm having a heart attack", "he had a cardiac arrest"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_group_3_stroke(self):
        for text in ["I think I'm having a stroke, my speech is slurred",
                     "slurred speech all of a sudden",
                     "her face has a facial droop",
                     "sudden arm weakness",
                     "I can't speak properly"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_group_4_breathing(self):
        for text in ["I can't breathe", "I feel like I'm suffocating", "she is choking"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_group_5_bleeding(self):
        for text in ["severe bleeding from my arm", "he is bleeding out"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_group_6_unconscious(self):
        for text in ["she is unconscious", "he passed out and won't wake",
                     "she fainted and is not waking up"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_group_6_fainted_alone_does_not_trigger(self):
        """'fainted' requires 'not waking' per the approved list."""
        self.assertIsNone(get_emergency_message("I fainted earlier but I'm fine now"))

    def test_group_7_seizure(self):
        for text in ["he is having a seizure", "she is convulsing",
                     "shaking uncontrollably"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_group_8_anaphylaxis(self):
        for text in ["I'm having an allergic reaction", "this could be anaphylaxis",
                     "my throat closing up"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_group_9_suicidal(self):
        for text in ["I want to die", "thinking about suicide",
                     "I might kill myself", "I've been feeling suicidal"]:
            with self.subTest(text=text):
                self.assertEqual(get_emergency_message(text), EMERGENCY_MESSAGE)

    def test_case_insensitive(self):
        self.assertEqual(get_emergency_message("I HAVE CHEST PAIN AND CAN'T BREATHE"),
                         EMERGENCY_MESSAGE)

    def test_message_text_is_exactly_the_approved_string(self):
        self.assertEqual(
            EMERGENCY_MESSAGE,
            "\U0001f6a8 This sounds like it could be a medical emergency. Please call your "
            "local emergency number (e.g., 911, 999, 112) or go to the nearest emergency "
            "department immediately. This tool is not designed for emergency situations.",
        )


class TestNonEmergencyInputs(unittest.TestCase):

    def test_normal_chief_complaints_do_not_trigger(self):
        for text in ["My leg is swollen",
                     "I have a sore throat",
                     "I have atrial fibrillation and want to check my stroke risk",
                     "my throat hurts when I swallow",
                     "there is a dent when I press my calf"]:
            with self.subTest(text=text):
                self.assertIsNone(get_emergency_message(text))


class TestGateHaltsApiFlow(unittest.TestCase):
    """End-to-end through the real endpoint: emergency message returned,
    no scoring, no routing, no state stored."""

    def setUp(self):
        conversations.clear()

    def test_emergency_returns_halt_marker(self):
        data = _post("I have chest pain and can't breathe", "emg_1")
        self.assertEqual(data.response, EMERGENCY_MESSAGE)
        self.assertEqual(data.type, "emergency")
        self.assertTrue(data.halt)
        self.assertIsNone(data.score_result)
        self.assertIsNone(data.doctor_report)

    def test_no_conversation_state_created(self):
        _post("I have chest pain and can't breathe", "emg_2")
        self.assertNotIn("emg_2", conversations)

    def test_emergency_mid_conversation_halts(self):
        """A trigger on any turn halts, even after a pathway has started."""
        _post("My leg is swollen", "emg_3")
        self.assertIn("emg_3", conversations)
        data = _post("actually I have chest pain and can't breathe", "emg_3")
        self.assertEqual(data.type, "emergency")
        self.assertTrue(data.halt)
        self.assertIsNone(data.score_result)

    def test_normal_flow_unaffected(self):
        data = _post("My leg is swollen", "emg_4")
        self.assertIsNone(data.type)
        self.assertFalse(data.halt)
        self.assertEqual(data.category, "leg_swelling")
        self.assertEqual(data.phase, "extraction")

    def test_stroke_trigger_message_from_spec(self):
        data = _post("I think I'm having a stroke, my speech is slurred", "emg_5")
        self.assertEqual(data.response, EMERGENCY_MESSAGE)
        self.assertEqual(data.type, "emergency")


if __name__ == "__main__":
    unittest.main(verbosity=2)
