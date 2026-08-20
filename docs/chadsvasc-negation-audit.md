# CHA₂DS₂-VASc Negation Coverage Audit (Stage 6, Part 1)

**Scope:** Audit `extract_chadsvasc_fields()` / `CHADSVASC_PATTERNS` in
`backend/extraction/rule_fallback.py` against the negation machinery already
built and tested for Wells'/Centor (the "ruled out" family, `but`
scope-termination, hedged-input handling — see
`docs/negation-pattern-diff-proposal.md` and
`backend/tests/test_negation_regression.py`). **No code changed in this
pass** — this is a report only, per Stage 6 instructions.

There is no separate `chadsvasc_extractor.py` file; the conversational layer
is `backend/extraction/afib_extractor.py`, which delegates all field
extraction to `extract_chadsvasc_fields()` in `rule_fallback.py` — the same
module that implements Wells'/Centor extraction and the shared negation
helpers (`_is_negated_context`, `_resolve_yes_no`, `RULED_OUT_POSSIBLE`,
`RULED_OUT_NEGATED`, `SCOPE_TERMINATORS`).

All findings below were reproduced against the current code (commands and
raw output available on request; representative outputs are inlined).

---

## 1. Background: two extraction code paths, only one gets the fixes

`extract_chadsvasc_fields()` (and, identically, `extract_wells_fields()` /
`extract_centor_fields()`) branches on whether the field being extracted is
the one the system just asked about:

```python
if last_asked_field == k:
    resolved = _resolve_yes_no(k, current_input)   # <-- "ruled out" family, hedging, scope-term all live here
    ...
# else: "opportunistic" path — field mentioned unprompted, or extracted
# retroactively from the running conversation
pats = CHADSVASC_PATTERNS[k]
if _positive_keyword_hit(combined, pats) and not cur_is_no and not _is_family_history(current_input):
    ...
```

`_resolve_yes_no()` — which contains the `RULED_OUT_POSSIBLE` /
`RULED_OUT_NEGATED` phrase checks and the hedged-input check — is **only
called from the `last_asked_field == k` branch**. The opportunistic branch
uses `_positive_keyword_hit()`, which only calls `_is_negated_context()`
(single-token `NEGATORS` scan + `but` scope-termination). It has **no
knowledge of the "ruled out" phrase family at all**.

This is an architectural gap shared by all three modules, not something
specific to CHA₂DS₂-VASc. It matters much more here because CHA₂DS₂-VASc's
fields (CHF, hypertension, stroke/TIA, vascular disease, diabetes) are
exactly the kind of history patients volunteer unprompted in an early,
free-text turn ("I have afib, high blood pressure, and my doctor ruled out
a stroke last year") — i.e., the opportunistic path is the common case for
this module, not the exception.

**Important correction to the premise implied by the Stage 6 request:**
"pseudo-negation guards" are not actually implemented anywhere in the
codebase yet. The original diff proposal (§3.1) proposed a
`PSEUDO_NEGATION_PHRASES` list, but the Stage 2c closing note only shipped
the "ruled out" family and `but` scope-termination — pseudo-negation
protection today is *incidental* (the 4-word window happens to be too short
to reach most DEEPEN-style long-distance false positives), not an explicit
guard. This audit treats "already fixed elsewhere" as meaning specifically
the ruled-out family and `but` scope-termination, since those are the only
two things actually shipped.

---

## 2. Per-concept findings

### 2.1 CHF history — **vulnerable (confirmed)**

Same sentence, two code paths, two different (wrong vs. right) answers:

| Input | Context | Result | Correct? |
|---|---|---|---|
| "Heart failure was ruled out." | `last_asked_field="chf_history"` | `chf_history=False` | ✅ correct |
| "By the way, heart failure was ruled out last year." | opportunistic (asked about hypertension instead) | `chf_history=True` | ❌ **wrong** — ruled out, should be `False` |

Root cause: opportunistic path has no "ruled out" phrase check, so "heart
failure" (a `YES_PATTERNS`/`CHADSVASC_PATTERNS` keyword) matches, and
`_is_negated_context()` doesn't flag "ruled out" as a negator (neither
"ruled" nor "out" is in `NEGATORS`).

### 2.2 Hypertension — **partially vulnerable**

| Input | Context | Result | Correct? |
|---|---|---|---|
| "Hypertension has not been ruled out." | last-asked | `unclear` | ✅ correct (via `RULED_OUT_POSSIBLE`) |
| "Hypertension has not been ruled out yet, by the way." | opportunistic | `unclear` | ✅ *accidentally* correct — falls through to the safe default because nothing matches, not because the double-negation is understood |
| "Hypertension is unlikely based on my last checkup." | last-asked | `hypertension=True` | ❌ **wrong** — "unlikely" is a possible-negation term (per original diff proposal §2.4) and was never implemented for *any* module; should be `unclear` |

### 2.3 Stroke / TIA history — **vulnerable, including a case NOT covered by opportunistic-vs-last-asked at all**

**(a) Same opportunistic-path gap as CHF**, using the exact sentence from
the existing Wells/Centor regression suite
(`test_negation_regression.py::test_ruled_out_stroke_is_false`), which only
tests this sentence via `last_asked_field="stroke_tia_history"`:

| Input | Context | Result | Correct? |
|---|---|---|---|
| "The doctor ruled out a stroke, so it must be something else." | `last_asked_field="stroke_tia_history"` | `stroke_tia_history=False` | ✅ correct |
| "The doctor ruled out a stroke, so it must be something else." | opportunistic (asked about diabetes instead) | `stroke_tia_history=True` | ❌ **wrong** — identical sentence, opposite answer depending only on which field the system happened to ask about last |

**(b) A distinct, deeper bug — confirmed broken even on the "already
fixed" last-asked path** — using the exact example phrase from the Stage 6
request:

> "No history of stroke, but I did have a TIA last year."

```
result, unclear = extract_chadsvasc_fields(
    text, text, ["stroke_tia_history"], last_asked_field="stroke_tia_history")
# -> {'stroke_tia_history': False}   (should be True)
```

This is **not** the opportunistic-path gap — it fails on the fully-fixed
`_resolve_yes_no()` path too. Cause: `stroke_tia_history` is a **compound
field** — both "stroke" and "tia" are `YES_PATTERNS`/`NO_PATTERNS` keywords
for the *same* field. The existing `but`-scope-termination fix
(`_no_pattern_scope_ends_before_yes`) only protects **cross-field**
leakage (e.g. "no cough" must not suppress a later, different field
"fever"). It is never invoked here because the `NO_PATTERNS` loop's literal
match ("no") gets skipped correctly (scope termination *does* fire and
skip that match, since "tia" appears after "but"), but the code then falls
through to the `YES_PATTERNS` loop, which matches **"stroke"** (not
"tia" — "stroke" comes first in the list and matches earlier in the
sentence). "stroke" is correctly detected as negated (the "no" is 3 words
before it, well within the 4-word window, and there's no scope terminator
*between* "no" and "stroke" — the "but" comes after). The function returns
`False` on this first match and never reaches "tia" later in the same
clause. In other words: negating one synonym inside a compound field
incorrectly negates the whole field, even when a *different* synonym for
the same field is affirmed later in the same sentence, on the far side of
a scope terminator.

This is a new failure mode not present in Wells/Centor, because those
modules don't have two distinct keyword-synonyms for one field appearing
together in a realistic sentence the way "stroke" and "TIA" naturally do
here (patients often mention both in one breath, one negated, one not).
`vascular_disease` has the same structural risk (heart attack / PAD /
aortic plaque, all one field) but no test phrase below happens to trigger
it exactly this way — worth testing before considering this fixed.

### 2.4 Vascular disease — **vulnerable to the opportunistic gap, plus a separate keyword-coverage gap**

| Input | Context | Result | Correct? | Note |
|---|---|---|---|---|
| "Vascular disease was ruled out after testing." | last-asked | `vascular_disease=False` | ✅ correct | matched via `_resolve_yes_no`'s YES_PATTERNS "heart attack"... wait — see below |
| "Vascular disease was ruled out after testing, apparently." | opportunistic | `unclear` | ⚠️ safe but for the wrong reason | **`CHADSVASC_PATTERNS["vascular_disease"]` does not contain the literal phrase "vascular disease"** — only "heart attack", "mi", "peripheral artery", "pad", "aortic(...)", "coronary", "stent", "bypass". So this sentence never even registers a keyword hit in the opportunistic path; it lands on `unclear` by coincidence, not because "ruled out" was understood. |
| "A heart attack has not been ruled out yet." | last-asked | `unclear` | ✅ correct (via `RULED_OUT_POSSIBLE`) | |
| "A heart attack has not been ruled out yet, they said." | opportunistic | `unclear` | ✅ *accidentally* correct | Falls through because the blunt single-token `not` (in `NEGATORS`) happens to sit within 4 words of "heart attack" and negates it via `_is_negated_context`, which coincidentally also produces "unclear" (not a wrongly-confident `True`) since it just fails to match \_positive_keyword_hit and there's no other signal. It is not using the "not ruled out = possible" semantic — it's a different, cruder mechanism landing on the same output. |

(Separately: `CHADSVASC_PATTERNS["vascular_disease"]` should probably also
include "vascular disease" and "peripheral artery disease" literally — a
non-negation pattern-coverage gap, noted for completeness but out of scope
for this negation audit.)

### 2.5 Diabetes — **vulnerable, same pattern as CHF**

| Input | Context | Result | Correct? |
|---|---|---|---|
| "Diabetes was ruled out last year." | last-asked | `diabetes=False` | ✅ correct |
| "Diabetes was ruled out last year, just so you know." | opportunistic | `diabetes=True` | ❌ **wrong** |
| "Diabetes is unlikely, according to my labs." | last-asked | `diabetes=True` | ❌ **wrong** — same "unlikely" gap as hypertension |

### 2.6 Age, sex category — **not a negation-family risk, but one related bug found**

- **Age** is extracted via `extract_age()` (numeric regex on
  "N years old" / "I'm N"). There is no boolean yes/no or "ruled out"
  semantic to negate — out of scope for this audit.
- **Sex category** *is* extracted via plain substring matching
  (`"female" in text`, checked before `"male"`), with **no negation
  awareness at all** — not part of the "ruled out" family the Stage 6
  request asked about, but a directly analogous bug worth flagging since
  it's the same root problem (naive keyword presence, no negation check)
  and it directly changes the score (+1 point for female):

  ```
  "I'm not a woman, I'm a man."   -> sex = 'female'   (WRONG, should be 'male')
  "I am a man, not a woman."      -> sex = 'female'   (WRONG, should be 'male')
  ```

  This reproduces in both `rule_fallback.extract_chadsvasc_fields()`'s sex
  branch and the near-duplicate sex-detection logic in
  `afib_extractor.extract_and_update_data()` (lines ~101-106), which has
  the identical substring-first-match structure. Flagging for awareness;
  not remediated in this pass since it's outside the requested "ruled
  out"/`but`/pseudo-negation scope, but it's a real, reproducible scoring
  bug and arguably higher priority than some of the negation gaps above
  since it silently flips a real point value rather than falling through
  to a safe "unclear."

### 2.7 Bonus finding: family-history exclusion is bypassed by direct keyword matches

Not part of the requested negation-family audit, but discovered while
tracing the same code paths and worth a one-line flag: `_is_family_history()`
is only checked in the last-asked branch *after* `_resolve_yes_no()` — but
`_resolve_yes_no()`'s `YES_PATTERNS` loop matches "heart failure" as a bare
keyword and returns `True` before the family-history check ever runs:

```
"My father had heart failure." (last_asked_field="chf_history")
-> chf_history=True   (WRONG — this is family history, not personal history)
```

CHA₂DS₂-VASc scores personal history only (see `_is_family_history()`'s own
docstring), so this is a real gap, but it's a different bug class
(keyword-vs-personal-history, not negation) — noted here for the record,
not addressed in this pass.

---

## 3. Summary table

| Concept | Opportunistic-path "ruled out" gap | Compound-field scope-termination gap | "Unlikely" gap | Other |
|---|---|---|---|---|
| CHF history | ❌ confirmed broken | n/a (single-concept field) | not tested | — |
| Hypertension | ✅ accidentally safe | n/a | ❌ confirmed broken | — |
| Stroke/TIA | ❌ confirmed broken | ❌ **confirmed broken even on the "fixed" path** | not tested | — |
| Vascular disease | ⚠️ accidentally safe, but only because of a missing keyword ("vascular disease" itself isn't a pattern) | not triggered by tested phrases, same structural risk as stroke/TIA | not tested | — |
| Diabetes | ❌ confirmed broken | n/a | ❌ confirmed broken | — |
| Age | n/a (numeric, not boolean) | n/a | n/a | — |
| Sex category | n/a (not part of "ruled out" family) | n/a | n/a | ❌ negation-blind substring match, directly affects score |

**Recommended priority for a future fix pass** (not done in this stage):
1. Stroke/TIA compound-field scope-termination bug (§2.3b) — breaks the
   exact example phrase requested in this task, on the already-"fixed" path.
2. Opportunistic-path "ruled out" gap, generalized across all four boolean
   CHA₂DS₂-VASc fields (§2.1, 2.3a, 2.5) — likely the single highest-value
   fix given how often patients volunteer this history unprompted.
3. Sex-category negation blindness (§2.6) — small phrase surface, but
   directly flips a scored point.
4. "Unlikely" possible-negation handling (§2.2, §2.5) — a known, long-
   standing gap from the original diff proposal, not chadsvasc-specific.

No fixes have been made. Stopping here for review per Stage 6 instructions.
