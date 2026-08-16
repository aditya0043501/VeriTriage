# Vague Input Scenarios — Sore Throat (Centor/McIsaac)

Manual verification scenarios for GAP 4. Extraction must map correctly or ask
a specific clarifying question — never guess silently.

## Scenario 1: "My throat is killing me and I feel gross"
- Expected: No criteria set from "gross". Clarifying questions about fever,
  cough, etc. begin naturally.

## Scenario 2: "I feel hot but I haven't taken my temperature"
- Expected: Agent may ask about chills/feeling feverish; "feeling feverish"
  can map fever = true per criteria, or agent suggests taking temperature.
  Must not demand a thermometer reading.

## Scenario 3: "I'm coughing a little, mostly at night"
- Expected: absence_of_cough = false (cough IS present). The double-negative
  criterion must be handled correctly.

## Scenario 4: "There's some white stuff back there I think"
- Expected: Mapped toward tonsillar_exudate = true, possibly after one
  confirming question (e.g., looked in a mirror?).

## Scenario 5: "I don't know what lymph nodes are"
- Expected: Agent rephrases in plain language ("sore lumps at the front of
  your neck when you press gently") rather than using the clinical term.

## Scenario 6: "It hurts to swallow, do I have strep??"
- Expected: Brief acknowledgment of concern; agent explains it will go through
  a few questions; no premature conclusion offered.

## Verification procedure
Same as chest pain scenarios: run each in a fresh conversation with a valid
API key, record extracted values and questions, PASS/FAIL per expectations.
