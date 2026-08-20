# Clinical Explanation Layer — Source Citations & Published Probability Data

This document is the traceability record for Stage 4 (patient-facing
explanation layer). Every risk-level label, probability percentage, or
"X% of people in this range" statement shown to a patient must trace back
to a number in this document, which in turn traces to a specific published
source. No probability number in the explanation layer may be invented,
interpolated, or estimated beyond what is explicitly reported below.

---

## 1. Wells' Criteria for DVT

### 1.1 Wells PS, Anderson DR, et al. Lancet. 1997;350:1795-8.
*"Value of assessment of pretest probability of deep-vein thrombosis in
clinical management."* PMID: 9428249. DOI: 10.1016/s0140-6736(97)08140-3

This is the original prospective management study (n=593) that validated
using the 3-tier Wells clinical probability model (low/moderate/high) to
decide how aggressively to pursue ultrasound testing, rather than as a
pure diagnostic/prevalence study. Reported DVT prevalence by pretest
category:

| Pretest category | DVT prevalence (this study) | n |
|---|---|---|
| Low | **3%** (10/329) | 329 |
| Moderate | **17%** (32/193) | 193 |
| High | **75%** (53/71) | 71 |

### 1.2 Wells PS, Owen C, Doucette S, Fergusson D, Tran H. JAMA. 2006;295:199-207.
*"Does this patient have deep vein thrombosis?"* PMID: 16403932.
DOI: 10.1001/jama.295.2.199

This is a systematic review pooling 14 studies (>8,000 patients) that
applied the Wells clinical prediction rule prior to D-dimer/imaging.
Because it pools many cohorts, this is the higher-confidence, larger-n
estimate, and is the one used by `probability_mapping.py` (§4 below) as
the primary cited figure for our low/moderate/high tiers:

| Wells score / clinical probability | DVT prevalence (pooled, 95% CI) |
|---|---|
| Low (score ≤ 0) | **5.0%** (95% CI 4.0%–8.0%) |
| Moderate (score 1–2) | **17%** (95% CI 13%–23%) |
| High (score ≥ 3) | **53%** (95% CI 44%–61%) |

**Discrepancy note (RESOLVED):** `backend/scoring/wells_score.py`
previously displayed the high-probability tier as "~17-53% DVT prevalence"
in its module docstring, and as the vague, number-free phrase "A
substantial share" in its patient-facing `full_text`. The "17-53%" figure
appeared to conflate the moderate-tier lower bound with the high-tier point
estimate; the correct cited number for the high tier per the 2006 JAMA
pooled data is **53%** (95% CI 44-61%), not a 17-53% range.

Neither location returned or displayed the literal "17-53%" string to a
patient — it existed only in the docstring (dead documentation, never
read or rendered) — but the live patient-facing text also carried no
specific number at all for this tier, which this document's table above
was written to correct for internal reference.

*Fix applied:* both locations were corrected.
- The docstring (line ~23) now reads: "Score >=3: High probability
  (~53% (44-61% CI), Wells 2006 JAMA pooled cohort)".
- The patient-facing `full_text` for the high tier now reads "About half
  of people in this range turn out to have a DVT..." — using the cited
  53% point estimate in plain language, without exposing the raw
  confidence interval to a layperson audience. The rest of the sentence
  (danger if untreated, immediate evaluation recommendation, "not a
  diagnosis" disclaimer) was left unchanged.

Both changes are text-only; no scoring logic changed, and the full 71-test
backend suite passed unchanged after the fix.

---

## 2. Modified Centor Score (McIsaac Modification)

### 2.1 Centor RM, Witherspoon JM, Dalton HP, Brody CE, Link K. Med Decis Making. 1981;1(3):239-46.
*"The diagnosis of strep throat in adults in the emergency room."*
PMID: 6763125. DOI: 10.1177/0272989X8100100304

This is the original Centor score derivation (adult ER patients only, no
age modifier — the 4-variable raw score: fever, absence of cough, tender
anterior cervical nodes, tonsillar exudate). Reported probability of
positive culture by raw score:

| Raw Centor score (0–4, no age adjustment) | Probability of positive culture |
|---|---|
| 0 | **2.5%** |
| 1 | **6.5%** |
| 2 | **15%** |
| 3 | **32%** |
| 4 | **56%** |

This raw-score table applies only to the original (adult, non-age-adjusted)
Centor score and is retained here for historical/completeness purposes. Our
system uses the McIsaac-modified score (§2.2), which includes the age term,
so this table is **not** used directly by `probability_mapping.py` — see
the note in §4.2.

### 2.2 McIsaac WJ, Kellner JD, Aufricht P, Vanjaka A, Low DE. JAMA. 2004;291(13):1587-1595.
*"Empirical validation of guidelines for the management of pharyngitis in
children and adults."* PMID: 15069046. DOI: 10.1001/jama.291.13.1587

This is the validation study for the age-adjusted "modified Centor" /
McIsaac score used by our `centor_score.py`. The per-score-level
probability-of-GAS-pharyngitis table below is the widely-cited clinical
reference table attributed to this validation (consistent with, e.g.,
Medscape's Centor Score (Modified) calculator and the Hong Kong Department
of Health's Acute Pharyngitis clinical guidance, both of which cite
McIsaac WJ et al., JAMA. 2004;291(13):1587 for this exact table):

| Modified (McIsaac) Centor score | Likelihood of streptococcal pharyngitis |
|---|---|
| ≤ 0 (i.e. -1 or 0) | **1% – 2.5%** |
| 1 | **5% – 10%** |
| 2 | **11% – 17%** |
| 3 | **28% – 35%** |
| 4 or 5 | **51% – 53%** |

**Consistency check against our current 3-tier consolidation:**
`backend/scoring/centor_score.py` currently consolidates this 5-level
table into 3 patient-facing tiers:
- Low (score ≤ 1) → displayed as "~1-10%" → consistent with merging the
  ≤0 (1-2.5%) and 1 (5-10%) rows above.
- Moderate (score 2-3) → displayed as "~11-35%" → consistent with merging
  the 2 (11-17%) and 3 (28-35%) rows above.
- High (score ≥ 4) → displayed as "~51%+" → consistent with the 4-or-5
  (51-53%) row above.

No discrepancy found for Centor (unlike Wells, §1.2). The existing 3-tier
text is a fair, source-consistent consolidation of the granular table.

### 2.3 Citation-pairing check (Stage 5): scoring citation vs. probability citation

There are, in total, **three** distinct Centor/McIsaac papers referenced
across this codebase, and it is easy to conflate them:

| Paper | Used for | Where |
|---|---|---|
| Centor RM, et al. Med Decis Making. 1981;1(3):239-46. | Original 4-criterion derivation (adult ER, no age term). Raw 0-4 score table (§2.1) — historical only, not used for any number shown to patients. | Half of `centor_score.py`'s `CITATION` constant |
| McIsaac WJ, et al. CMAJ. 1998;158(1):75-83. | Original derivation of the age-adjusted ("modified Centor") scoring *rule itself* (which age brackets get +1/0/-1). | Other half of `centor_score.py`'s `CITATION` constant — this is the "Scoring criteria source" shown in the mockups |
| McIsaac WJ, et al. JAMA. 2004;291(13):1587-1595. | Later, larger validation study that produced the per-score-level *probability* table used in §2.2. | `probability_mapping.MCISAAC_2004_JAMA_CITATION` — this is the "Probability estimate source" shown in the mockups |

**Finding:** `calculate_centor_score()`'s returned `citation` field (Centor
1981 + McIsaac 1998 CMAJ) is a **different pair of papers** from the one
`get_centor_probability()` cites (McIsaac 2004 JAMA). This is the same
citation-splitting pattern already used for Wells (§1.2: criteria cited to
Wells 2003 NEJM, probability cited to Wells 2006 JAMA) and is intentional,
not an error — the point-scoring rule and the probability-of-strep table
are genuinely two different published claims, so they legitimately draw on
two different studies. The mockups (`centor_mockup_demo.py`) label these
explicitly as "Scoring criteria source" and "Probability estimate source"
on separate lines, precisely so the two citations don't read as
inconsistent restatements of the same fact.

**What would be a real bug (checked for, not present):** substituting the
*raw*, non-age-adjusted 1981 Centor table (§2.1: 2.5%/6.5%/15%/32%/56% for
raw scores 0-4) for the McIsaac age-adjusted score's probability, or vice
versa. Because the modified score's range (-1 to 5) differs from the raw
score's range (0-4), and a raw score of e.g. 4 does not mean the same thing
as a modified score of 4 once the age term is folded in, these two tables
are **not interchangeable**. Confirmed: `probability_mapping.py` only ever
reads from the McIsaac 2004 JAMA table (§2.2) and never from the 1981 raw
table; `centor_score.py`'s own tier thresholds (§2.2 consistency check
above) also only use the age-adjusted score. No cross-contamination found.

---

## 3. What this means for the explanation layer

- Every probability shown to a patient must come from the tables in §1.2
  (Wells, pooled 2006 JAMA numbers) or §2.2 (Centor, McIsaac 2004 JAMA
  table), not from the 1997 Lancet or 1981 Centor tables, which are
  historical/context citations only.
- If a computed score ever falls outside the ranges tabulated above (should
  not happen given the fixed point ranges of these scores, but included as
  a safety check), `probability_mapping.py` must reject the request rather
  than guess or interpolate — see §4.

## 4. Probability-mapping function behavior

Implemented in `backend/explanations/probability_mapping.py`.

- `get_wells_probability(score: int) -> ProbabilityResult`
  - score ≤ 0 → 5.0% (95% CI 4.0-8.0%), cite Wells 2006 JAMA
  - 1 ≤ score ≤ 2 → 17% (95% CI 13-23%), cite Wells 2006 JAMA
  - score ≥ 3 → 53% (95% CI 44-61%), cite Wells 2006 JAMA
  - Any non-integer or otherwise malformed input → rejected (raises
    `UncitedScoreError`), never guessed.

- `get_centor_probability(score: int) -> ProbabilityResult`
  - score ≤ 0 → 1%-2.5%, cite McIsaac 2004 JAMA
  - score == 1 → 5%-10%, cite McIsaac 2004 JAMA
  - score == 2 → 11%-17%, cite McIsaac 2004 JAMA
  - score == 3 → 28%-35%, cite McIsaac 2004 JAMA
  - score ≥ 4 → 51%-53%, cite McIsaac 2004 JAMA
  - Any score outside the mathematically possible range for this score
    (i.e. below -1 or above 5, given the scoring formula in
    `centor_score.py`) → rejected (raises `UncitedScoreError`).
