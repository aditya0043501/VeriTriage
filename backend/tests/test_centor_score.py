"""
Test suite for Centor/McIsaac score calculation
Hand-calculated test cases based on published criteria
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scoring.centor_score import calculate_centor_score


def test_centor_classic_presentation():
    """22yo female, 2-day sore throat, feels feverish (38.6C), no cough,
    white patches on tonsils, tender lumps on front of neck.
    Fever(1) + no cough(1) + tender nodes(1) + exudate(1) + age 22 (0) = 4 -> high."""
    result = calculate_centor_score(
        fever=True, absence_of_cough=True,
        tender_cervical_nodes=True, tonsillar_exudate=True, age=22
    )
    assert result["score"] == 4, f"Expected 4, got {result['score']}"
    assert result["tier"] == "high", f"Expected high, got {result['tier']}"
    assert result["isPartial"] == True
    assert "Rapid strep test / throat culture" in result["pendingFields"]
    print("✓ test_centor_classic_presentation: PASS")


def test_centor_age_over_45_subtracts():
    """Same symptoms at age 50: 4 - 1 = 3 -> moderate (McIsaac age >=45 is -1)."""
    result = calculate_centor_score(
        fever=True, absence_of_cough=True,
        tender_cervical_nodes=True, tonsillar_exudate=True, age=50
    )
    assert result["score"] == 3, f"Expected 3, got {result['score']}"
    assert result["tier"] == "moderate", f"Expected moderate, got {result['tier']}"
    assert result["breakdown"]["Age"]["points"] == -1
    print("✓ test_centor_age_over_45_subtracts: PASS")


def test_centor_low_risk():
    """Cough present, no fever, no nodes, no exudate, age 30: 0 points -> low."""
    result = calculate_centor_score(
        fever=False, absence_of_cough=False,
        tender_cervical_nodes=False, tonsillar_exudate=False, age=30
    )
    assert result["score"] == 0, f"Expected 0, got {result['score']}"
    assert result["tier"] == "low", f"Expected low, got {result['tier']}"
    print("✓ test_centor_low_risk: PASS")


def test_centor_negative_score():
    """No criteria, age 60: 0 - 1 = -1 -> low (score can go negative)."""
    result = calculate_centor_score(
        fever=False, absence_of_cough=False,
        tender_cervical_nodes=False, tonsillar_exudate=False, age=60
    )
    assert result["score"] == -1, f"Expected -1, got {result['score']}"
    assert result["tier"] == "low", f"Expected low, got {result['tier']}"
    print("✓ test_centor_negative_score: PASS")


def test_centor_moderate_boundary():
    """Fever + no cough, age 30: 2 points -> moderate."""
    result = calculate_centor_score(
        fever=True, absence_of_cough=True,
        tender_cervical_nodes=False, tonsillar_exudate=False, age=30
    )
    assert result["score"] == 2, f"Expected 2, got {result['score']}"
    assert result["tier"] == "moderate", f"Expected moderate, got {result['tier']}"
    print("✓ test_centor_moderate_boundary: PASS")


def test_centor_output_structure():
    """Output must match the standardized structure."""
    result = calculate_centor_score(
        fever=True, absence_of_cough=False,
        tender_cervical_nodes=False, tonsillar_exudate=False, age=25
    )
    assert isinstance(result["recommendation"], dict)
    for field in ["what_to_do", "who_to_see", "how_soon", "full_text"]:
        assert field in result["recommendation"], f"Missing {field}"
    for label, entry in result["breakdown"].items():
        assert "points" in entry and "justification" in entry, f"Bad breakdown entry: {label}"
    assert "Centor" in result["citation"] and "McIsaac" in result["citation"]
    print("✓ test_centor_output_structure: PASS")


if __name__ == "__main__":
    print("Running Centor/McIsaac score test suite...")
    print()
    test_centor_classic_presentation()
    test_centor_age_over_45_subtracts()
    test_centor_low_risk()
    test_centor_negative_score()
    test_centor_moderate_boundary()
    test_centor_output_structure()
    print()
    print("All tests passed! ✓")
