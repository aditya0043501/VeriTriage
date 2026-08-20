# Negation-Handling Sources & Citations

This document records the research basis for VeriTriage's negation-handling
approach. The deterministic extractors do not implement NegEx itself; rather,
the patterns below informed the design of our trigger-term lists,
pseudo-negation handling, and scope-termination heuristics.

---

## 1. Chapman et al. 2001 — NegEx

**Citation:** Chapman WW, Bridewell W, Hanbury P, Cooper GF, Buchanan BG.
"NegEx — A Simple Algorithm for Identifying Negated Findings and Diseases
in Discharge Summaries." *Journal of Biomedical Informatics*. 2001;34(5):301-310.
DOI: [10.1006/jbin.2001.1029](https://doi.org/10.1006/jbin.2001.1029)

NegEx is the foundational rule-based algorithm for detecting negated clinical
findings in free text. It introduces the idea of ranking trigger phrases into
negating, pseudo-negating, and possibility categories, then scanning a fixed
window forward from each trigger to decide whether a target concept falls
within the negated scope. Our extractors borrow this three-tier structure —
true negation, pseudo-negation, and uncertainty — even though we apply it to
layperson symptom language rather than discharge summaries.

---

## 2. Original NegEx Repository

**Source:** https://github.com/chapmanbe/negex

The repository ships the canonical trigger-phrase lists and a set of
annotated example sentences that we consulted when building our own pattern
sets for phrases like "no history of", "denies", and "rules out". Reading the
annotated sentences helped us calibrate how far a negation scope should
reasonably extend in short clinical utterances, and which trigger terms tend
to co-occur with the findings we extract (cough, stroke, DVT signs, etc.).

---

## 3. negspacy

**Source:** https://github.com/jenojp/negspacy

negspacy is a spaCy pipeline component that wraps NegEx-style logic and, more
usefully for us, documents a curated set of pseudo-negation terms
(e.g. "no evidence of ... developing") and scope-termination cues. We used its
pseudo-negation and termination token lists as a cross-check against our own
`PSEUDO_NEGATIONS` and scope-end heuristics to make sure we were not
over-negating phrases such as "no concern for ... but I do have".

---

## 4. DEEPEN — PMC5863758

**Citation:** Peng Y, Gupta S, Mydulam A, et al. "DEEPEN: De-identification,
Extraction, and Evaluation Pipeline for Enhanced clinical Notes."
*PMC5863758.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5863758/

DEEPEN reports NegEx's known false-positive failure modes on complex
sentences, particularly where a negation trigger and the target finding are
separated by intervening clauses or where pseudo-negation is misread as true
negation. These documented weaknesses motivated our pattern-level
improvements — specifically the combined-text greedy-match guard that
prevents premature field extraction from general conversational context, and
the stricter scope we apply when a sentence contains both a negation and a
later affirmative mention of the same finding.

---

## 5. "Beyond Negation Detection" — arXiv 2503.17425

**Citation:** "Beyond Negation Detection: ...". arXiv:2503.17425.
https://arxiv.org/abs/2503.17425

This paper benchmarks NegEx against GPT-4o on clinical negation tasks and
reports per-category error rates, giving us a methodology reference for the
GPT-vs-deterministic comparison in our own `benchmark/` directory. It also
informs how we frame the trade-off in our documentation: rule-based
extractors are fast, deterministic, and private, but LLMs handle syntactic
ambiguity better — which is exactly the gap our pattern refinements aim to
narrow without introducing a model dependency.

---

*This is a research-basis document only. It does not imply that VeriTriage
implements NegEx, negspacy, or any of the cited systems. Patterns were
adapted, not copied.*
