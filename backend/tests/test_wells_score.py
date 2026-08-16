"""
Test suite for Wells' DVT score calculation
Hand-calculated test cases based on published criteria
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scoring.wells_score import calculate_wells_score


def test_wells_post_surgical_patient():
    """58yo male, hip surgery 5 days ago (day 6 post-op), new right calf swelling
    and pain. Patient-knowable criteria: recent major surgery (1), localized
    tenderness (1), calf swelling (1) -> 3 points -> high probability.
    (His shortness of breath / fast heart rate warrant urgent care regardless.)"""
    result = calculate_wells_score(
        bedridden_or_surgery=True,
        localized_tenderness=True,
        calf_swelling_over_3cm=True
    )
    assert result["score"] == 3, f"Expected 3, got {result['score']}"
    assert result["tier"] == "high", f"Expected high, got {result['tier']}"
    assert result["isPartial"] == True
    assert "D-dimer / ultrasound" in result["pendingFields"]
    assert "Alternative diagnosis assessment" in result["pendingFields"]
    print("✓ test_wells_post_surgical_patient: PASS")


def test_wells_zero_score():
    """No criteria present -> 0 points -> low probability"""
    result = calculate_wells_score()
    assert result["score"] == 0, f"Expected 0, got {result['score']}"
    assert result["tier"] == "low", f"Expected low, got {result['tier']}"
    print("✓ test_wells_zero_score: PASS")


def test_wells_moderate():
    """Two criteria -> 2 points -> moderate probability"""
    result = calculate_wells_score(entire_leg_swollen=True, pitting_edema=True)
    assert result["score"] == 2, f"Expected 2, got {result['score']}"
    assert result["tier"] == "moderate", f"Expected moderate, got {result['tier']}"
    print("✓ test_wells_moderate: PASS")


def test_wells_boundary_1_point():
    """One criterion -> 1 point -> moderate (1-2 band per published cutoffs)"""
    result = calculate_wells_score(active_cancer=True)
    assert result["score"] == 1, f"Expected 1, got {result['score']}"
    assert result["tier"] == "moderate", f"Expected moderate, got {result['tier']}"
    print("✓ test_wells_boundary_1_point: PASS")


def test_wells_all_criteria():
    """All 8 criteria -> 8 points -> high probability"""
    result = calculate_wells_score(
        active_cancer=True, paralysis_or_immobilization=True,
        bedridden_or_surgery=True, localized_tenderness=True,
        entire_leg_swollen=True, calf_swelling_over_3cm=True,
        pitting_edema=True, collateral_veins=True
    )
    assert result["score"] == 8, f"Expected 8, got {result['score']}"
    assert result["tier"] == "high", f"Expected high, got {result['tier']}"
    print("✓ test_wells_all_criteria: PASS")


def test_wells_output_structure():
    """Output must match the standardized structure with justifications and 3-part recommendation"""
    result = calculate_wells_score(active_cancer=True)
    assert isinstance(result["recommendation"], dict)
    for field in ["what_to_do", "who_to_see", "how_soon", "full_text"]:
        assert field in result["recommendation"], f"Missing {field}"
    for label, entry in result["breakdown"].items():
        assert "points" in entry and "justification" in entry, f"Bad breakdown entry: {label}"
    assert "Wells" in result["citation"]
    print("✓ test_wells_output_structure: PASS")


if __name__ == "__main__":
    print("Running Wells' DVT score test suite...")
    print()
    test_wells_post_surgical_patient()
    test_wells_zero_score()
    test_wells_moderate()
    test_wells_boundary_1_point()
    test_wells_all_criteria()
    test_wells_output_structure()
    print()
    print("All tests passed! ✓")
