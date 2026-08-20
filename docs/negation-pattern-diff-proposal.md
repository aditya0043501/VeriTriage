# Negation Pattern Expansion — Diff Proposal (Stage 2)

**Scope:** Cross-check the original NegEx trigger list against VeriTriage's
current extraction patterns, and propose additions *without* editing the
production tables.

**Current pattern tables reviewed:**
- `backend/extraction/rule_fallback.py`
  - `NEGATORS` (single-token negation scan)
  - `YES_PATTERNS` / `NO_PATTERNS`
  - `WELLS_PATTERNS`, `CENTOR_PATTERNS`, `CHADSVASC_PATTERNS`
  - `_is_negated_context()` window-based scan
- `backend/extraction/extraction_utils.py`
  - `UNCERTAINTY_MARKERS` / `_is_hedged()`
  - `FAMILY_HISTORY_MARKERS`

**Source list compared:**
- `negex.python/negex_triggers.txt` from https://github.com/chapmanbe/negex
  (191 trigger lines, after expanding tag labels).

---

## 1. How our current logic already covers many NegEx triggers

Our `_is_negated_context()` is token-based and window-limited, so the
single-word NegEx triggers are largely covered by the current `NEGATORS`
set:

| NegEx trigger token | Covered by `NEGATORS`? | Notes |
|---|---|---|
| `no`, `not`, `never`, `none`, `nor`, `neither` | yes | Core set. |
| `without` | yes | Catches most `with no …`/`without …` forms. |
| `cannot`, `can't` | yes | `cannot` token present. |
| `free` | yes | Catches `free of`, `cough-free`, etc. |
| `absent` / `absence of` | partial | `absent` token present; `absence of` phrase is **not** caught ("absence" alone is not in the set). |
| `not` + verb (`not appear`, `not have`, etc.) | yes | `not` token catches these. |
| `no` + noun (`no evidence`, `no sign of`, etc.) | yes | `no` token catches these. |
| `never had`, `never developed` | yes | `never` token catches these. |

Because the window is **4 words**, our implementation already avoids many of
the long-distance false-positive failure modes documented by DEEPEN (e.g.
"no evidence of extension of his infected **pseudocyst** into the psoas
muscle" — the negation token is well outside the 4-word window of the target).
That is a feature, not a bug.

---

## 2. NegEx triggers NOT currently covered

For each uncovered trigger, the table gives:
- The exact NegEx line and tag.
- A patient-facing example sentence in our domain.
- The proposed VeriTriage category.

### 2.1 Pseudo-negation / false-trigger terms (do **not** negate the target condition)

| NegEx trigger | Example in our domain | Proposed category | Rationale |
|---|---|---|---|
| `no increase` | "There has been **no increase** in my calf swelling." | pseudo-negation/false-trigger | The swelling still exists; "no increase" negates a change, not the condition. |
| `no change` | "**No change** in my sore throat today." | pseudo-negation/false-trigger | Condition still present. |
| `no significant change` | "**No significant change** in the swelling." | pseudo-negation/false-trigger | Same as above; also a DEEPEN-style false-positive pattern. |
| `no interval change` | "**No interval change** in my symptoms." | pseudo-negation/false-trigger | Condition still present. |
| `no definite change` | "**No definite change** in my blood sugar." | pseudo-negation/false-trigger | Condition still present. |
| `no suspicious change` | "**No suspicious change** in the CT report." | pseudo-negation/false-trigger | Condition may still be present; "suspicious" is the negated property. |
| `not extend` | "The clot does **not extend** into my thigh." | pseudo-negation/false-trigger | Negates spatial extent, not the presence of the clot. |
| `not cause` | "The fall did **not cause** the swelling." | pseudo-negation/false-trigger | Negates causality, not the condition. |
| `not drain` | "The abscess did **not drain**." | pseudo-negation/false-trigger | Negates an action, not the abscess itself. |
| `gram negative` | "I was told the culture was **gram negative**." | pseudo-negation/false-trigger | Clinical micro term; "negative" here is not negation. |
| `without difficulty` | "I can walk **without difficulty**." | pseudo-negation/false-trigger | Affirms normal function; does not negate a finding. |
| `not only` | "**Not only** do I have chest pain, but my leg is swollen." | pseudo-negation/false-trigger | Focus particle, not negation of the following condition. |
| `not certain if` | "I'm **not certain if** I have diabetes." | possible/hedged | Uncertainty marker → should route to clarification. |
| `not certain whether` | "I'm **not certain whether** it was a stroke." | possible/hedged | Uncertainty marker. |
| `not necessarily` | "It's **not necessarily** heart failure." | possible/hedged | Uncertainty. |

### 2.2 Pre-negation terms (negation appears before the target condition)

| NegEx trigger | Example in our domain | Proposed category | Rationale |
|---|---|---|---|
| `absence of` | "**Absence of** fever." | pre-negation | Direct negation; token `absent` exists but the phrase `absence of` is not token-matched. |
| `denied` / `denies` / `denying` | "I **deny** any chest pain." / "My chart **denies** hypertension." | pre-negation | Common in patient self-report; currently missed because neither word is in `NEGATORS`. |
| `declined` / `declines` | "I **declined** the diabetes test." | possible/hedged | Refusal of testing ≠ absence of condition; better treated as unclear. |
| `fails to reveal` | "The ultrasound **fails to reveal** a clot." | pre-negation | Implies the target was not found; rare in patient speech. |
| `negative for` | "The blood test was **negative for** diabetes." | pre-negation | Direct result negation; currently missed. |
| `no complaints of` | "I have **no complaints of** chest pain." | pre-negation | Direct negation; already caught by `no`, but phrase-aware matching would be cleaner. |
| `no cause of` | "They found **no cause of** the fever." | pseudo-negation/false-trigger | The fever exists; only its cause is negated. |
| `resolved` | "The swelling has **resolved**." | post-negation | Past-present condition now absent; appears after the keyword in natural word order. |
| `ruled out` family (PREN forms: `rules out`, `did rule out`, `can rule out`, `adequate to rule out`, `sufficient to rule out`) | "The doctor **rules out** a stroke." / "The workup **ruled out** DVT." | pre- or post-negation | Strong completed negation. Must be phrase-matched; currently missed because `rules`/`ruled`/`out` are not in `NEGATORS`. |
| `unremarkable for` | "The exam was **unremarkable for** heart failure." | pre-negation | Clinical-document language; unlikely in patient speech. |
| `evaluate for` / `test for` / `to exclude` | "I came in to **evaluate for** a blood clot." | possible/hedged | Workup language; does not confirm absence. |

### 2.3 Post-negation terms (negation appears after the target condition)

| NegEx trigger | Example in our domain | Proposed category | Rationale |
|---|---|---|---|
| `was ruled out` / `is ruled out` / `are ruled out` / `have been ruled out` / `has been ruled out` | "Diabetes **was ruled out** last year." | post-negation | Strong completed negation; currently missed. |

### 2.4 Possible-negation terms (condition is under consideration, not confirmed absent)

| NegEx trigger | Example in our domain | Proposed category | Rationale |
|---|---|---|---|
| `unlikely` | "A stroke is **unlikely**." | possible/hedged | Probability term; should route to clarification, not strong negative. |
| `rule out` / `r/o` / `ro` | "We need to **rule out** a stroke." | possible/hedged | Plan or differential; not a result. |
| `be ruled out for` / `should be ruled out for` / `ought to be ruled out for` / `may be ruled out for` / `might be ruled out for` / `could be ruled out for` / `will be ruled out for` / `can be ruled out for` / `must be ruled out for` / `is to be ruled out for` / `what must be ruled out is` | "A heart attack **must be ruled out**." | possible/hedged | Still under evaluation. |

### 2.5 Scope-terminating conjunctions (NegEx `[CONJ]` family)

Our `_is_negated_context()` currently has **no scope terminator**. A leading
negation token can bleed across a conjunction and flip the wrong field.

| NegEx trigger | Example showing the failure | Proposed category |
|---|---|---|
| `but` | "I have **no** cough, **but** I **do** have a fever." | conjunction/scope-termination |
| `however` | "No headache, **however** my throat hurts." | conjunction/scope-termination |
| `yet` | "No fever **yet**, but my neck glands are swollen." | conjunction/scope-termination |
| `although` / `though` | "**Although** I have no cough, I do have a fever." | conjunction/scope-termination |
| `nevertheless` / `still` | "No chest pain, **still** my leg is swollen." | conjunction/scope-termination |
| `aside from` / `except` / `apart from` | "No issues **except** the leg swelling." | conjunction/scope-termination |
| `secondary to` / `as the cause/source/reason/etiology/origin of/for` / `cause of/for` / `source of/for` / `reason of/for` / `etiology of/for` / `origin of/for` / `trigger event for` | "**Blood clot** was mentioned as a concern, but the swelling is **secondary to** the sprain." (attempted crossing case — see closing note below) | conjunction/scope-termination |

> **Status update (Stage 2c-close): `but` implemented and tested; `secondary to`
> deprioritized.**
>
> `but` scope termination was implemented in `rule_fallback.py` (both in the
> `NO_PATTERNS` scan and in `_is_negated_context()`'s window scan) and is
> covered by regression tests in `backend/tests/test_negation_regression.py`.
>
> `secondary to` was **not** implemented. Across two rounds of attempting to
> construct a genuine patient-facing scope-crossing test case, no realistic
> sentence was found where a negator incorrectly crosses "secondary to" to
> reach a target concept:
> - The original candidate ("The leg swelling is secondary to the sprain,
>   **not** a blood clot.") turned out to be ordinary adjacent negation —
>   "not" directly negates "blood clot" with no crossing involved, so `True`
>   (negated) is the clinically correct answer, not a bug.
> - A second candidate ("Blood clot was mentioned as a concern, **but** the
>   swelling is secondary to the sprain.") contains no negation trigger at
>   all, so there's nothing for "secondary to" to incorrectly pull a negation
>   across.
>
> This pattern appears more relevant to clinician-note language (where
> "no operative intervention ... secondary to X" constructions are common,
> per the DEEPEN paper) than to patient free-text. **Scope termination for
> "secondary to" is deprioritized — not implemented** — pending a real
> example from actual patient input data (e.g. beta-test transcripts) that
> demonstrates the failure mode.

### 2.6 Post-possible negation / scope terminators that *preserve* the target

| NegEx trigger | Example in our domain | Proposed category | Rationale |
|---|---|---|---|
| `did not rule out` / `not ruled out` / `not been ruled out` / `being ruled out` / `be ruled out` / `should/may/might/could/will/can/must be ruled out` (when the condition precedes the phrase) | "The doctor **did not rule out** a stroke." | possible/hedged | Negation of a negation = condition remains possible. |

---

## 3. Suggested implementation shape (for review)

No files were edited. The following is the *shape* of the change I would make
if approved.

### 3.1 Option A: phrase-aware negation helper (preferred)

Replace the purely token-based `_is_negated_context()` with a helper that
matches multi-word pre- and post-negation phrases, plus a small set of
pseudo-negation and scope-termination patterns. For example:

```python
PRE_NEGATION_PHRASES = [
    "absence of", "denies", "denied", "denying",
    "negative for", "fails to reveal", "rules out", "ruled out",
    "did rule out", "can rule out", "adequate to rule out",
    "sufficient to rule out",
]

POST_NEGATION_PHRASES = [
    "was ruled out", "is ruled out", "are ruled out",
    "have been ruled out", "has been ruled out",
    "resolved",
]

PSEUDO_NEGATION_PHRASES = [
    "no increase", "no change", "no significant change",
    "no interval change", "no definite change", "no suspicious change",
    "not extend", "not cause", "not drain", "not only",
    "gram negative", "without difficulty", "no cause of",
]

SCOPE_TERMINATORS = [
    "but", "however", "nevertheless", "yet", "though", "although", "still",
    "aside from", "except", "apart from", "secondary to",
    # causal/as-source phrases
    "as the cause of", "as the source of", "as the reason of",
    "as the etiology of", "as the origin of",
    "as the cause for", "as the source for", "as the reason for",
    "as the etiology for", "as the origin for",
    "as a cause of", "as a source of", "as a reason of",
    "as a etiology of", "as a cause for", "as a source for",
    "as a reason for", "as a etiology for",
    "as an cause of", "as an source of", "as an reason of",
    "as an etiology of", "as an origin of",
    "as an cause for", "as an source for", "as an reason for",
    "as an etiology for", "as an origin for",
    "cause of", "cause for", "causes of", "causes for",
    "source of", "source for", "sources of", "sources for",
    "reason of", "reason for", "reasons of", "reasons for",
    "etiology of", "etiology for", "trigger event for",
    "origin of", "origin for", "origins of", "origins for",
    "other possibilities of",
]
```

The helper would:
1. Locate the keyword.
2. Search the 4-word window for any **pseudo-negation** phrase first; if
   found, return `False` (not negated).
   - *Priority rule:* pseudo-negation is checked before scope terminators
     because a pseudo-negation phrase is more specific than any shorter
     terminator substring it contains (e.g. "no cause of" contains "cause of";
     the pseudo-negation reading must win).
3. Search the window for a **pre-negation** phrase ending before the keyword,
   or a **post-negation** phrase starting after the keyword; if found, return
   `True`.
4. If a **scope terminator** appears between the negation phrase and the
   keyword, terminate the scope and return `False`.
5. Fall back to the existing single-token `NEGATORS` scan for forms like
   "not", "no", "never", etc.

**Window policy:** phrase-aware matching operates **inside the existing
4-word window**, not in a separate larger window. This preserves the
DEEPEN-protection property of the current token scan. The trade-off is that
long pseudo-/pre-negation phrases may consume the entire window before
they reach the target concept; those phrases are listed below in §3.1.1.

#### 3.1.1 Proposed phrases at risk of never firing inside a 4-word window

The following proposed phrases are unlikely to match the keyword when the
phrase itself fills most or all of the 4-word span. They are still listed
for completeness, but we should expect them to land as `unclear` unless the
keyword appears very close to the trigger.

| Proposed phrase | Target example | Why it may not fire |
|---|---|---|
| `no evidence of extension of the swollen calf` | calf swelling fields | "no evidence of extension" (4 words) ends exactly at the window edge; "swollen calf" starts 3 words later, outside the before-window. |
| `no significant change in my tonsillar swelling` | tonsillar_exudate | "no significant change" (3 words) plus "in my tonsillar" pushes the keyword past the 4-word before-window. |
| `what must be ruled out is` | any condition | 6-word phrase; far exceeds the 4-word window. |
| `sufficient/adequate to rule out` | any condition | 4-word phrase consumes the entire after-window; the condition must appear immediately before the trigger. |

Phrases that **do** fit inside the 4-word window and are expected to fire
reliably include: `denies`, `negative for`, `rules out`, `ruled out`,
`was ruled out`, `no cause of the fever`, `no change in swelling`,
`but`, `however`, `secondary to`, and most 2-3 word conjunctions.

### 3.2 Option B: targeted token additions (minimal)

Add only the high-impact single tokens to `NEGATORS` and `UNCERTAINTY_MARKERS`:

```python
NEGATORS |= {"denies", "denied", "denying", "resolved"}
UNCERTAINTY_MARKERS += [
    "unlikely", "rule out", "ruled out?", "being ruled out",
    "not ruled out", "not been ruled out", "did not rule out",
]
```

This is smaller but does **not** fix the multi-word forms (`rules out`,
`was ruled out`, etc.) or the scope-termination problem.

---

## 4. DEEPEN false-positive overlap with our 153 test phrases

DEEPEN's known false-positive patterns are:
1. Long-distance negation where the negation token modifies an intervening noun,
   not the target condition.
2. "No [significant/interval/new/other] [evidence/change] in [condition]."
3. "No [complications/extensions/intervention] of [condition]."
4. "No [evidence] of [noun] consistent with [condition]."
5. "[Condition] was ruled out" vs. "[Condition] is not ruled out" confusion.

### 4.1 Direct overlap with our test phrases

Searching the 153-phrase set for these structural patterns:

| Pattern | Found in our 153 phrases? | Matching phrase(s) | Our current handling |
|---|---|---|---|
| `no evidence of [condition]` | **No** | — | Not tested. |
| `no significant interval change in [condition]` | **No** | — | Not tested. |
| `no [complications/extensions] of [condition]` | **No** | — | Not tested. |
| `no [evidence] of [noun] consistent with [condition]` | **No** | — | Not tested. |
| `[condition] was ruled out` | **No** | — | Not tested. |
| `ruled out [condition]` | **No** | — | Not tested. |
| `unlikely` | **No** | — | Not tested. |
| Long negation (>4 words) before target | **Limited** | "never diagnosed with heart failure" (chf_history) | Handled: `never` is within 4 words. |

**Conclusion:** Our existing 153 test phrases do **not** directly exercise the
DEEPEN long-distance false-positive cases. The test set is heavy on short,
explicit negations (`no stroke history`, `never had a TIA`, `no diabetes`,
`not hypertensive`, etc.) that are well covered by the current token set.

### 4.2 Phrases that would be *vulnerable* if we expanded the negation window

These are phrases in our test set that are currently handled correctly
*because* the window is short. If we ever widened the window to match more
NegEx-style phrases, these would become false positives:

| Field | Phrase | Why vulnerable if window widens |
|---|---|---|
| `fever` | "not hot or shivery" | Multi-word scope; works only because `not` is adjacent. |
| `absence_of_cough` | "I haven't coughed once" | Works because `haven't` is within 1 word. |
| `chf_history` | "never diagnosed with heart failure" | `never` is within 3 words; a 6-word window would still catch it, but a dependency-style long-distance match could mis-attach `never` to other verbs. |
| `vascular_disease` | "no stents or bypass" | Works because `no` is adjacent. |

These are not DEEPEN false positives today, but they show why any
expansion must stay window-limited or add scope terminators.

### 4.3 DEEPEN-style cases we should add to the test set

If we add the proposed patterns, the following candidate phrases should be
added to `benchmark/test_phrases.py` (or a dedicated negation-regression
suite) to make sure DEEPEN-style failures stay fixed:

| Proposed test phrase | Expected | DEEPEN failure mode it exercises |
|---|---|---|
| "No evidence of extension of the swollen calf into the thigh." | unclear / condition still present | Negation modifies intervening noun, not target. |
| "No significant change in my tonsillar swelling." | unclear / still present | "No change" pseudo-negation. |
| "The doctor ruled out a stroke, so it must be something else." | `stroke_tia_history` = False | Strong completed post-negation. |
| "A stroke has not been ruled out yet." | unclear / possible | Double-negation: condition remains possible. |
| "I have no cough, but I do have a fever." | `fever` = True | Scope terminator across a conjunction. |
| "The leg swelling is secondary to the sprain, not a blood clot." | `active_cancer`/DVT criterion stays as-is, not flipped by "not" | Scope terminator prevents `not` from crossing `secondary to`. |

---

## 5. Summary and recommendation

**Already well covered:**
- Short, explicit patient negations (`no`, `not`, `never`, `none`, `without`,
  `cannot`, `free`, `absent`).
- The 4-word window largely prevents DEEPEN-style long-distance false
  positives.

**Gaps to close:**
1. **Phrase-level negation** for `denies`, `ruled out`, `negative for`,
   `resolved`, `absence of`.
2. **Scope termination** across conjunctions (`but`, `however`, `although`,
   `secondary to`, causal `as … of/for` phrases).
3. **Pseudo-negation filtering** for `no change`, `no increase`, `no cause of`,
   etc.
4. **Possible-negation handling** for `rule out`, `must be ruled out`,
   `unlikely`, `not ruled out`.
5. **Test coverage** for DEEPEN-style sentences is missing from the current
   153-phrase set.

**Proposed next step:** Implement Option A (phrase-aware helper) in
`rule_fallback.py`, then add the DEEPEN-style regression phrases from §4.3 to
the benchmark set.

**No production files were changed in this stage.**
