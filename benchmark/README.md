# VeriTriage LLM Comparison Benchmark

**This is NOT part of the VeriTriage product.** It is a one-time benchmark
script used to generate comparison data for a pitch deck. The product itself
remains 100% OpenAI-free — this folder is never imported by `main.py`,
extractors, or any production code.

## What it does

Compares VeriTriage's deterministic rule-based extraction against a generic
LLM (OpenAI GPT-4o-mini) on the same 153 adversarial test phrases, measuring:

- Overall accuracy (%)
- Accuracy on the "hard" subset (negation, hedging, family-history cases)
- Non-determinism rate (LLM run 3× on hedged phrases; VeriTriage is always 0%)

## Setup

```bash
cd benchmark
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your OpenAI API key (gitignored, never committed):
echo "OPENAI_API_KEY=sk-..." > .env

# Run the benchmark:
python llm_comparison.py
```

## Output

- `benchmark_results.md` — markdown summary table for the pitch deck
- `benchmark_results.json` — full raw data (every phrase, both systems' answers)

## Cost

~153 + 51 (3× hedged repeats) = ~204 API calls to GPT-4o-mini.
At current pricing this is well under $0.10 total.

## Cleanup after benchmark

```bash
# Remove the venv and results (keep the script for reproducibility):
rm -rf venv benchmark_results.json
# benchmark_results.md can be kept for the pitch deck
```
