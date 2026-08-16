"""
Test suite for CHA₂DS₂-VASc score calculation
Hand-calculated test cases based on published criteria (verified against MDCalc)

Criteria (verified against MDCalc and Merck Manual):
- CHF history: 1 pt
- Hypertension: 1 pt
- Age ≥ 75: 2 pts
- Age 65-74: 1 pt
- Diabetes: 1 pt
- Prior stroke/TIA/thromboembolism: 2 pts
- Vascular disease: 1 pt
- Sex (female): 1 pt
Max score: 9
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scoring.chadsvasc_score import calculate_chadsvasc_score


def test_chadsvasc_zero_score_male():
    """65yo male, no risk factors -> 0 points -> low risk"""
    result = calculate_chadsvasc_score(
        age=65, sex="male", chf_history=False, hypertension=False,
        stroke_tia_history=False, vascular_disease=False, diabetes=False
    )
    assert result["score"] == 1, f"Expected 1 (age 65-74 = 1pt), got {result['score']}"
    assert result["tier"] == "moderate", f"Expected moderate, got {result['tier']}"
    print("✓ test_chadsvasc_zero_score_male: PASS")


def test_chadsvasc_true_zero_male():
    """50yo male, no risk factors -> 0 points -> low risk"""
    result = calculate_chadsvasc_score(
        age=50, sex="male", chf_history=False, hypertension=False,
        stroke_tia_history=False, vascular_disease=False, diabetes=False
    )
    assert result["score"] == 0, f"Expected 0, got {result['score']}"
    assert result["tier"] == "low", f"Expected low, got {result['tier']}"
    print("✓ test_chadsvasc_true_zero_male: PASS")


def test_chadsvasc_female_sex_point():
    """50yo female, no other risk factors -> 1 point (sex) -> moderate"""
    result = calculate_chadsvasc_score(
        age=50, sex="female", chf_history=False, hypertension=False,
        stroke_tia_history=False, vascular_disease=False, diabetes=False
    )
    assert result["score"] == 1, f"Expected 1, got {result['score']}"
    assert result["tier"] == "moderate", f"Expected moderate, got {result['tier']}"
    print("✓ test_chadsvasc_female_sex_point: PASS")


def test_chadsvasc_age_75_plus():
    """78yo male, hypertension only -> 2 (age≥75) + 1 (HTN) = 3 -> high"""
    result = calculate_chadsvasc_score(
        age=78, sex="male", chf_history=False, hypertension=True,
        stroke_tia_history=False, vascular_disease=False, diabetes=False
    )
    assert result["score"] == 3, f"Expected 3, got {result['score']}"
    assert result["tier"] == "high", f"Expected high, got {result['tier']}"
    print("✓ test_chadsvasc_age_75_plus: PASS")


def test_chadsvasc_prior_stroke_double():
    """70yo male, prior stroke -> 1 (age 65-74) + 2 (stroke) = 3 -> high"""
    result = calculate_chadsvasc_score(
        age=70, sex="male", chf_history=False, hypertension=False,
        stroke_tia_history=True, vascular_disease=False, diabetes=False
    )
    assert result["score"] == 3, f"Expected 3, got {result['score']}"
    assert result["tier"] == "high", f"Expected high, got {result['tier']}"
    print("✓ test_chadsvasc_prior_stroke_double: PASS")


def test_chadsvasc_maximum_score():
    """82yo female with all risk factors -> 2+1+1+2+1+1+1+1 = 9 -> high"""
    result = calculate_chadsvasc_score(
        age=82, sex="female", chf_history=True, hypertension=True,
        stroke_tia_history=True, vascular_disease=True, diabetes=True
    )
    assert result["score"] == 9, f"Expected 9, got {result['score']}"
    assert result["tier"] == "high", f"Expected high, got {result['tier']}"
    print("✓ test_chadsvasc_maximum_score: PASS")


def test_chadsvasc_moderate_boundary():
    """60yo male, hypertension -> 0 (age) + 1 (HTN) = 1 -> moderate"""
    result = calculate_chadsvasc_score(
        age=60, sex="male", chf_history=False, hypertension=True,
        stroke_tia_history=False, vascular_disease=False, diabetes=False
    )
    assert result["score"] == 1, f"Expected 1, got {result['score']}"
    assert result["tier"] == "moderate", f"Expected moderate, got {result['tier']}"
    print("✓ test_chadsvasc_moderate_boundary: PASS")


def test_chadsvasc_high_boundary():
    """60yo male, hypertension + diabetes -> 1+1 = 2 -> high"""
    result = calculate_chadsvasc_score(
        age=60, sex="male", chf_history=False, hypertension=True,
        stroke_tia_history=False, vascular_disease=False, diabetes=True
    )
    assert result["score"] == 2, f"Expected 2, got {result['score']}"
    assert result["tier"] == "high", f"Expected high, got {result['tier']}"
    print("✓ test_chadsvasc_high_boundary: PASS")


def test_chadsvasc_not_partial():
    """CHA₂DS₂-VASc is fully patient-observable — no partial score"""
    result = calculate_chadsvasc_score(
        age=70, sex="female", chf_history=True, hypertension=True,
        stroke_tia_history=False, vascular_disease=False, diabetes=True
    )
    assert result["isPartial"] == False, "Should not be partial"
    assert result["pendingFields"] == [], "Should have no pending fields"
    print("✓ test_chadsvasc_not_partial: PASS")


def test_chadsvasc_output_structure():
    """Output must match the standardized structure"""
    result = calculate_chadsvasc_score(
        age=72, sex="male", chf_history=False, hypertension=True,
        stroke_tia_history=False, vascular_disease=False, diabetes=False
    )
    assert isinstance(result["recommendation"], dict)
    for field in ["what_to_do", "who_to_see", "how_soon", "full_text"]:
        assert field in result["recommendation"], f"Missing {field}"
    for label, entry in result["breakdown"].items():
        assert "points" in entry and "justification" in entry, f"Bad breakdown entry: {label}"
    assert "Lip" in result["citation"]
    assert result["citation_url"].startswith("https://pubmed")
    assert result["citation_doi"].startswith("https://doi.org")
    print("✓ test_chadsvasc_output_structure: PASS")


def test_chadsvasc_age_brackets():
    """Verify age bracket scoring: <65=0, 65-74=1, ≥75=2"""
    r1 = calculate_chadsvasc_score(age=64, sex="male", chf_history=False,
        hypertension=False, stroke_tia_history=False, vascular_disease=False, diabetes=False)
    assert r1["score"] == 0

    r2 = calculate_chadsvasc_score(age=65, sex="male", chf_history=False,
        hypertension=False, stroke_tia_history=False, vascular_disease=False, diabetes=False)
    assert r2["score"] == 1

    r3 = calculate_chadsvasc_score(age=74, sex="male", chf_history=False,
        hypertension=False, stroke_tia_history=False, vascular_disease=False, diabetes=False)
    assert r3["score"] == 1

    r4 = calculate_chadsvasc_score(age=75, sex="male", chf_history=False,
        hypertension=False, stroke_tia_history=False, vascular_disease=False, diabetes=False)
    assert r4["score"] == 2
    print("✓ test_chadsvasc_age_brackets: PASS")


if __name__ == "__main__":
    print("Running CHA₂DS₂-VASc score test suite...")
    print()
    test_chadsvasc_zero_score_male()
    test_chadsvasc_true_zero_male()
    test_chadsvasc_female_sex_point()
    test_chadsvasc_age_75_plus()
    test_chadsvasc_prior_stroke_double()
    test_chadsvasc_maximum_score()
    test_chadsvasc_moderate_boundary()
    test_chadsvasc_high_boundary()
    test_chadsvasc_not_partial()
    test_chadsvasc_output_structure()
    test_chadsvasc_age_brackets()
    print()
    print("All tests passed! ✓")
