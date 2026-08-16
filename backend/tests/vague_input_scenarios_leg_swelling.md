# Vague Input Scenarios — Leg Swelling (Wells' DVT)

Manual verification scenarios for GAP 4. Extraction must map correctly or ask
a specific clarifying question — never guess silently.

## Scenario 1: "My leg feels heavy and puffy"
- Expected: Clarifying question — whole leg or one part? Which leg? No criteria
  set from "puffy" alone.

## Scenario 2: "I was in the hospital a while back"
- Expected: Clarifying question about how recently and whether it involved
  surgery or being bedridden >3 days. "A while back" must not be silently
  mapped to the 12-week surgery window.

## Scenario 3: "It hurts when I poke my calf, like a bruise"
- Expected: Mapped toward localized tenderness = true, possibly after one
  confirming question about where exactly it hurts.

## Scenario 4: "I don't know if one leg is bigger, maybe?"
- Expected: Agent suggests a plain comparison (look at both calves side by side)
  OR treats as not confirmed (false) and moves on. Never demands a measurement.

## Scenario 5: "I'm freaking out, my dad died of a blood clot"
- Expected: Brief acknowledgment of the fear first. Family history is NOT a
  Wells' criterion — it must not be counted. Agent continues with actual criteria.

## Scenario 6: "When I press it, my finger leaves a dent"
- Expected: pitting_edema = true (plain-language description correctly parsed).

## Verification procedure
Same as chest pain scenarios: run each in a fresh conversation with a valid
API key, record extracted values and questions, PASS/FAIL per expectations.
