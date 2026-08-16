# VeriTriage — Known Limitations (for testers)

**This is a prototype, not medical advice.** It uses validated clinical
scoring systems (Wells, Centor/McIsaac, CHA₂DS₂-VASc) but the scores
are based on what you self-report — they are not a diagnosis.

## What you're testing

A pre-visit intake tool that asks you questions about your symptoms and
produces a risk score + a summary you could bring to a doctor. There are
three pathways:

1. **Sore throat** — say "I have a sore throat" (or similar)
2. **Leg swelling** — say "my leg is swollen" (or similar)
3. **Atrial fibrillation / stroke risk** — say "I have atrial fibrillation"
   (or similar)

## Known limitations

- **State resets if the server restarts.** Conversations are stored in
  memory. If the server cycles (free tier sleeps after inactivity), your
  conversation is lost. Just start a new one.

- **Free tier cold starts.** The first request after inactivity may take
  10-30 seconds to respond. Subsequent requests are fast.

- **No persistence.** There's no database. Refreshing the page starts a
  new conversation. There's no login or account.

- **Not validated for everyone.** If you say you're under 18, pregnant,
  or immunocompromised, the tool will stop and tell you to see a doctor
  directly — the scores aren't validated for those populations.

- **Partial scores.** Wells and Centor scores are marked "PARTIAL" because
  some criteria require in-person testing (blood tests, throat swabs,
  ultrasound). The tool scores only what you can report at home.

- **Extraction isn't perfect.** If you use unusual phrasing, the tool may
  ask you to clarify. This is by design — it would rather ask again than
  guess wrong. If it asks for clarification, try rephrasing or use the
  Yes/No/Not sure buttons.

- **One conversation at a time per browser tab.** Each tab gets its own
  conversation ID. Multiple people can use the tool simultaneously without
  interfering with each other.

## What to test

1. **Happy path:** Start a conversation for each of the 3 pathways above.
   Answer the questions honestly (or make up a scenario). See if the
   score and recommendation make sense.

2. **Free text:** Try answering in your own words instead of using the
   buttons. e.g., "Yeah I've been burning up" instead of just "Yes."

3. **Negation:** Try saying "no" in different ways — "I don't have that,"
   "never had it," "nope." See if the tool correctly records it as negative.

4. **Hedging:** Try saying "maybe," "I think so," "not sure." The tool
   should ask you to clarify rather than guessing.

5. **Category switch:** Start a sore throat conversation, then say "actually
   my leg is swollen." The tool should offer to switch.

6. **Edge cases:** Say you're 15, or pregnant, or on chemo. The tool should
   stop and redirect you.

## How to give feedback

Just tell me:
- What pathway you tested
- What you said
- What the tool did (right or wrong)
- Any confusing or broken behavior

Screenshots welcome. No bug report too small.
