"""
Extraction robustness test suite for all 3 modules.
Tests the deterministic rule-based extractors against 15-20 varied
real-world phrasing scenarios with hand-verified ground truth.

These tests verify that the extractors correctly handle:
- Direct yes/no answers
- Negation-aware extraction ("no I don't have cancer" -> False)
- Descriptive language ("my leg is swollen and tender" -> localized_tenderness=True)
- Vague but valid input ("since yesterday", "it hurts a lot")
- Compound answers answering multiple fields at once
- Bare "no" attributed to the last asked field
- Varied phrasings for the same clinical concept
- Unclear / low-confidence input -> marked unclear
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extraction.rule_fallback import (
    extract_wells_fields, extract_centor_fields, extract_chadsvasc_fields,
    detect_yes_no, extract_age, _acknowledge_input,
    WELLS_PATTERNS, CENTOR_PATTERNS, CHADSVASC_PATTERNS,
)


# ============================================================
# Wells' DVT — extraction robustness
# ============================================================

class TestWellsExtraction:

    def test_direct_yes_to_cancer_question(self):
        """Patient says 'yes I'm being treated for cancer' -> active_cancer=True"""
        result, unclear = extract_wells_fields("", "yes I'm being treated for cancer", ["active_cancer"], last_asked_field="active_cancer")
        assert result.get("active_cancer") == True
        assert "active_cancer" not in unclear

    def test_negation_aware_cancer(self):
        """Patient says 'no I don't have cancer' -> active_cancer=False (not True)"""
        result, unclear = extract_wells_fields("", "no I don't have cancer", ["active_cancer"], last_asked_field="active_cancer")
        assert result.get("active_cancer") == False
        assert "active_cancer" not in unclear

    def test_descriptive_tenderness(self):
        """Opening description mentions tenderness -> localized_tenderness=True"""
        combined = "my right leg is swollen and it's tender when I press on the calf"
        result, unclear = extract_wells_fields(combined, "my right leg is swollen and it's tender when I press on the calf",
                                       ["localized_tenderness"])
        assert result.get("localized_tenderness") == True

    def test_entire_leg_swollen_phrasings(self):
        """Various phrasings for entire leg swollen"""
        for phrase in ["the whole leg is swollen", "my entire leg is swollen", "all of the leg is swollen"]:
            result, unclear = extract_wells_fields(phrase, phrase, ["entire_leg_swollen"])
            assert result.get("entire_leg_swollen") == True, f"Failed for: {phrase}"

    def test_no_surgery_negation(self):
        """'no surgery' -> bedridden_or_surgery=False"""
        result, unclear = extract_wells_fields("", "no surgery", ["bedridden_or_surgery"], last_asked_field="bedridden_or_surgery")
        assert result.get("bedridden_or_surgery") == False

    def test_pitting_edema_descriptive(self):
        """'it leaves a dent when I press' -> pitting_edema=True"""
        combined = "when I press my finger into the swelling it leaves a dent"
        result, unclear = extract_wells_fields(combined, combined, ["pitting_edema"])
        assert result.get("pitting_edema") == True

    def test_bare_no_attribution(self):
        """Bare 'no' should be handled by last_asked_field in the extractor, not here.
        Here we just verify detect_yes_no returns False for bare 'no'."""
        assert detect_yes_no("no") is False
        assert detect_yes_no("nope") is False
        assert detect_yes_no("yes") is True

    def test_active_cancer_varied_phrasings(self):
        """Multiple ways of saying yes/no to active cancer"""
        for phrase in ["yes", "yeah", "yep", "I have cancer", "I'm being treated for cancer", "I do"]:
            result, unclear = extract_wells_fields("", phrase, ["active_cancer"], last_asked_field="active_cancer")
            assert result.get("active_cancer") == True, f"Failed for: {phrase}"
        for phrase in ["no", "nope", "I don't have cancer", "never", "not really"]:
            result, unclear = extract_wells_fields("", phrase, ["active_cancer"], last_asked_field="active_cancer")
            assert result.get("active_cancer") == False, f"Failed for: {phrase}"

    def test_paralysis_varied_phrasings(self):
        """Multiple ways of saying yes/no to paralysis/immobilization"""
        for phrase in ["yes", "I have a cast", "my leg is in a splint", "I can't move it"]:
            result, unclear = extract_wells_fields("", phrase, ["paralysis_or_immobilization"], last_asked_field="paralysis_or_immobilization")
            assert result.get("paralysis_or_immobilization") == True, f"Failed for: {phrase}"
        for phrase in ["no", "nope", "I don't have any", "no cast", "no splint"]:
            result, unclear = extract_wells_fields("", phrase, ["paralysis_or_immobilization"], last_asked_field="paralysis_or_immobilization")
            assert result.get("paralysis_or_immobilization") == False, f"Failed for: {phrase}"

    def test_collateral_veins_varied_phrasings(self):
        """Multiple ways of describing new visible veins"""
        for phrase in ["yes I see new veins", "there are visible surface veins", "I noticed spider veins"]:
            result, unclear = extract_wells_fields("", phrase, ["collateral_veins"], last_asked_field="collateral_veins")
            assert result.get("collateral_veins") == True, f"Failed for: {phrase}"
        for phrase in ["no new veins", "I don't see any", "no"]:
            result, unclear = extract_wells_fields("", phrase, ["collateral_veins"], last_asked_field="collateral_veins")
            assert result.get("collateral_veins") == False, f"Failed for: {phrase}"

    def test_unclear_input_returns_unclear(self):
        """A deliberately ambiguous answer is marked unclear, not guessed"""
        result, unclear = extract_wells_fields("", "maybe", ["active_cancer"], last_asked_field="active_cancer")
        assert result.get("active_cancer") is None
        assert "active_cancer" in unclear


# ============================================================
# Centor / McIsaac — extraction robustness
# ============================================================

class TestCentorExtraction:

    def test_fever_varied_phrasings(self):
        """Various fever phrasings should all extract fever=True"""
        for phrase in ["I have a fever", "I'm feverish", "my temperature is 101",
                       "I feel hot and shivery", "I've been burning up"]:
            result, unclear = extract_centor_fields(phrase, phrase, ["fever"])
            assert result.get("fever") == True, f"Failed for: {phrase}"

    def test_no_cough_means_absence_true(self):
        """'no cough' -> absence_of_cough=True"""
        result, unclear = extract_centor_fields("", "no I don't have a cough", ["absence_of_cough"])
        assert result.get("absence_of_cough") == True

    def test_cough_present_means_absence_false(self):
        """'yes I have a cough' -> absence_of_cough=False"""
        result, unclear = extract_centor_fields("", "yes I have a cough", ["absence_of_cough"])
        assert result.get("absence_of_cough") == False

    def test_swollen_glands_phrasings(self):
        """Various phrasings for tender cervical nodes"""
        for phrase in ["I have swollen neck glands", "tender lumps on my neck",
                       "swollen lumps near my jaw", "my neck glands are swollen"]:
            result, unclear = extract_centor_fields(phrase, phrase, ["tender_cervical_nodes"])
            assert result.get("tender_cervical_nodes") == True, f"Failed for: {phrase}"

    def test_white_patches_phrasings(self):
        """Various phrasings for tonsillar exudate"""
        for phrase in ["white patches on my tonsils", "white spots in the back of my throat",
                       "there's white stuff on my tonsils"]:
            result, unclear = extract_centor_fields(phrase, phrase, ["tonsillar_exudate"])
            assert result.get("tonsillar_exudate") == True, f"Failed for: {phrase}"

    def test_age_extraction_varied(self):
        """Age extraction from various phrasings"""
        assert extract_age("I'm 30 years old") == 30
        assert extract_age("I am 45") == 45
        assert extract_age("I'm 28") == 28
        assert extract_age("no specific age") is None

    def test_negation_aware_fever(self):
        """'no fever' -> fever=False (not True from keyword match)"""
        result, unclear = extract_centor_fields("", "no fever", ["fever"], last_asked_field="fever")
        assert result.get("fever") == False

    def test_fever_yes_no_variants(self):
        """Multiple yes/no ways to answer fever question"""
        for phrase in ["yes", "yeah", "yep", "I do", "I have one", "sure", "that's right"]:
            result, unclear = extract_centor_fields("", phrase, ["fever"], last_asked_field="fever")
            assert result.get("fever") == True, f"Failed for: {phrase}"
        for phrase in ["no", "nope", "I don't", "not really", "I do not"]:
            result, unclear = extract_centor_fields("", phrase, ["fever"], last_asked_field="fever")
            assert result.get("fever") == False, f"Failed for: {phrase}"
        # Hedged/vague answers should remain unclear rather than be guessed
        result, unclear = extract_centor_fields("", "I don't think so", ["fever"], last_asked_field="fever")
        assert result.get("fever") is None and "fever" in unclear, "Hedged answer should be unclear"

    def test_cough_yes_no_variants(self):
        """Multiple ways of saying yes/no to cough question"""
        for phrase in ["yes I have a cough", "yeah I cough", "I am coughing"]:
            result, unclear = extract_centor_fields("", phrase, ["absence_of_cough"])
            assert result.get("absence_of_cough") == False, f"Failed for: {phrase}"
        for phrase in ["no cough", "I don't have a cough", "not coughing"]:
            result, unclear = extract_centor_fields("", phrase, ["absence_of_cough"])
            assert result.get("absence_of_cough") == True, f"Failed for: {phrase}"

    def test_tonsil_yes_no_variants(self):
        """Multiple ways to answer tonsillar exudate question"""
        for phrase in ["yes", "I see white patches", "there are white spots", "I have them"]:
            result, unclear = extract_centor_fields("", phrase, ["tonsillar_exudate"], last_asked_field="tonsillar_exudate")
            assert result.get("tonsillar_exudate") == True, f"Failed for: {phrase}"
        for phrase in ["no", "no white patches", "I don't see any"]:
            result, unclear = extract_centor_fields("", phrase, ["tonsillar_exudate"], last_asked_field="tonsillar_exudate")
            assert result.get("tonsillar_exudate") == False, f"Failed for: {phrase}"
        # Descriptive "normal" phrasing should also resolve to False
        result, unclear = extract_centor_fields("", "my tonsils look normal", ["tonsillar_exudate"], last_asked_field="tonsillar_exudate")
        assert result.get("tonsillar_exudate") == False, f"Failed for: my tonsils look normal"

    def test_unclear_centor_input(self):
        """Ambiguous Centor answer returns unclear"""
        result, unclear = extract_centor_fields("", "sort of", ["fever"], last_asked_field="fever")
        assert result.get("fever") is None
        assert "fever" in unclear


# ============================================================
# CHA₂DS₂-VASc — extraction robustness
# ============================================================

class TestChadsvascExtraction:

    def test_hypertension_phrasings(self):
        """Various hypertension phrasings"""
        for phrase in ["I have high blood pressure", "yes I have hypertension",
                       "my blood pressure is high"]:
            result, unclear = extract_chadsvasc_fields(phrase, phrase, ["hypertension"])
            assert result.get("hypertension") == True, f"Failed for: {phrase}"

    def test_negation_aware_hypertension(self):
        """'no high blood pressure' -> hypertension=False"""
        result, unclear = extract_chadsvasc_fields("", "no I don't have high blood pressure", ["hypertension"], last_asked_field="hypertension")
        assert result.get("hypertension") == False

    def test_diabetes_phrasings(self):
        """Various diabetes phrasings"""
        for phrase in ["I have diabetes", "yes I'm diabetic", "I have type 2 diabetes",
                       "I take metformin for my blood sugar"]:
            result, unclear = extract_chadsvasc_fields(phrase, phrase, ["diabetes"])
            assert result.get("diabetes") == True, f"Failed for: {phrase}"

    def test_stroke_history_phrasings(self):
        """Various stroke/TIA history phrasings"""
        for phrase in ["I had a stroke", "yes I had a mini-stroke",
                       "I've had a TIA before"]:
            result, unclear = extract_chadsvasc_fields(phrase, phrase, ["stroke_tia_history"])
            assert result.get("stroke_tia_history") == True, f"Failed for: {phrase}"

    def test_negation_aware_stroke(self):
        """'no stroke' -> stroke_tia_history=False"""
        result, unclear = extract_chadsvasc_fields("", "no stroke", ["stroke_tia_history"], last_asked_field="stroke_tia_history")
        assert result.get("stroke_tia_history") == False

    def test_sex_extraction(self):
        """Sex extraction from various phrasings"""
        assert extract_chadsvasc_fields("I'm female", "I'm female", ["sex"])[0].get("sex") == "female"
        assert extract_chadsvasc_fields("I'm a woman", "I'm a woman", ["sex"])[0].get("sex") == "female"
        assert extract_chadsvasc_fields("I'm male", "I'm male", ["sex"])[0].get("sex") == "male"
        assert extract_chadsvasc_fields("I'm a man", "I'm a man", ["sex"])[0].get("sex") == "male"

    def test_vascular_disease_phrasings(self):
        """Various vascular disease phrasings"""
        for phrase in ["I had a heart attack", "yes I have peripheral artery disease",
                       "I've had a stent put in"]:
            result, unclear = extract_chadsvasc_fields(phrase, phrase, ["vascular_disease"])
            assert result.get("vascular_disease") == True, f"Failed for: {phrase}"

    def test_chf_phrasings(self):
        """Various CHF phrasings"""
        for phrase in ["I have heart failure", "yes I have congestive heart failure",
                       "I've been told I have CHF"]:
            result, unclear = extract_chadsvasc_fields(phrase, phrase, ["chf_history"])
            assert result.get("chf_history") == True, f"Failed for: {phrase}"

    def test_chadsvasc_yes_no_variants(self):
        """Multiple yes/no ways to answer CHA₂DS₂-VASc criteria"""
        for field in ["chf_history", "hypertension", "stroke_tia_history", "vascular_disease", "diabetes"]:
            for phrase in ["yes", "yeah", "yep", "I do", "I have", "I have it"]:
                result, unclear = extract_chadsvasc_fields("", phrase, [field], last_asked_field=field)
                assert result.get(field) == True, f"Failed for field={field} phrase={phrase}"
            for phrase in ["no", "nope", "I don't", "not really", "I do not", "never"]:
                result, unclear = extract_chadsvasc_fields("", phrase, [field], last_asked_field=field)
                assert result.get(field) == False, f"Failed for field={field} phrase={phrase}"

    def test_unclear_chadsvasc_input(self):
        """Ambiguous CHA₂DS₂-VASc answer returns unclear"""
        result, unclear = extract_chadsvasc_fields("", "unsure", ["diabetes"], last_asked_field="diabetes")
        assert result.get("diabetes") is None
        assert "diabetes" in unclear


# ============================================================
# Shared utilities — yes/no detection robustness
# ============================================================

class TestYesNoDetection:

    def test_clear_yes(self):
        assert detect_yes_no("yes") is True
        assert detect_yes_no("yeah") is True
        assert detect_yes_no("yep") is True
        assert detect_yes_no("that's right") is True
        assert detect_yes_no("correct") is True

    def test_clear_no(self):
        assert detect_yes_no("no") is False
        assert detect_yes_no("nope") is False
        assert detect_yes_no("not really") is False
        assert detect_yes_no("i don't") is False

    def test_uncertain_returns_none(self):
        assert detect_yes_no("maybe") is None
        assert detect_yes_no("i don't know") is None
        assert detect_yes_no("not sure") is None

    def test_negation_prefix(self):
        """'no I don't have diabetes' should be False, not True from 'I have'"""
        assert detect_yes_no("no I don't have diabetes") is False
        assert detect_yes_no("no cancer") is False


# ============================================================
# Acknowledge input — vague input handling
# ============================================================

class TestAcknowledgeInput:

    def test_timing_acknowledgment(self):
        assert "timing" in _acknowledge_input("since yesterday").lower()

    def test_pain_acknowledgment(self):
        assert "uncomfortable" in _acknowledge_input("it hurts a lot").lower()

    def test_yes_acknowledgment(self):
        assert "thank" in _acknowledge_input("yes").lower()

    def test_no_acknowledgment(self):
        assert "understood" in _acknowledge_input("no").lower()

    def test_uncertain_acknowledgment(self):
        assert "okay" in _acknowledge_input("I don't know").lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
