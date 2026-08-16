"""
Simple test script to verify the backend API works.
Uses only deterministic extraction — no LLM calls.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from router import classify_complaint, get_out_of_scope_message
from extraction import LegSwellingData, SoreThroatData, AFibStrokeData
from scoring import calculate_wells_score, calculate_centor_score, calculate_chadsvasc_score

print("Testing VeriTriage Backend Components...\n")

# Test 1: Out of scope message
print("Test 1: Out of Scope Message")
print("-" * 40)
print(get_out_of_scope_message())
print()

# Test 2: Router classification
print("Test 2: Router Classification")
print("-" * 40)
for text, expected in [
    ("my leg is swollen", "leg_swelling"),
    ("I have a sore throat", "sore_throat"),
    ("I have atrial fibrillation", "afib_stroke"),
    ("I have a headache", "out_of_scope"),
    ("I'm not feeling okay", "vague"),
]:
    result = classify_complaint(text)
    status = "PASS" if result == expected else f"FAIL (got {result})"
    print(f"  {text!r} -> {result} [{status}]")
print()

# Test 3: Data structures
print("Test 3: Data Structures")
print("-" * 40)
for name, cls in [("LegSwelling", LegSwellingData), ("SoreThroat", SoreThroatData), ("AFibStroke", AFibStrokeData)]:
    data = cls()
    print(f"{name}: complete={data.is_complete()}, missing={data.get_missing_fields()}")
print()

# Test 4: Scoring functions
print("Test 4: Scoring Functions")
print("-" * 40)
wells = calculate_wells_score(active_cancer=True, localized_tenderness=True)
print(f"Wells: score={wells['score']}, tier={wells['tier']}")

centor = calculate_centor_score(fever=True, absence_of_cough=True, tender_cervical_nodes=True, tonsillar_exudate=True, age=30)
print(f"Centor: score={centor['score']}, tier={centor['tier']}")

chadsvasc = calculate_chadsvasc_score(age=72, sex="male", chf_history=False, hypertension=True, stroke_tia_history=False, vascular_disease=False, diabetes=False)
print(f"CHA₂DS₂-VASc: score={chadsvasc['score']}, tier={chadsvasc['tier']}")
print()

print("All backend component tests completed!")
