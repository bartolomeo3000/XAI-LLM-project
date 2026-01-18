# Quick Start Guide: LLM Judge

## Setup (One-time)

```bash
# Set your API key for the judge LLM
export LLM_API_KEY="your-api-key-here"

# Optional: Set custom API endpoint (defaults to OpenAI)
export LLM_BASE_URL="https://api.openai.com/v1"
```

## Basic Usage

### Evaluate existing answers:

```bash
python run_llm_judge.py \
    --questions data/questions.jsonl \
    --answers runs/answers_20260112_223619.jsonl \
    --output runs/judgments_20260112_223619.jsonl
```

### With custom judge model:

```bash
python run_llm_judge.py \
    --questions data/questions.jsonl \
    --answers runs/answers_20260112_223619.jsonl \
    --output runs/judgments_20260112_223619.jsonl \
    --judge-model gpt-4o \
    --judge-temperature 0.0
```

### Full workflow:

```bash
# 1. Generate answers with your model
python run_prompting.py \
    --questions data/questions.jsonl \
    --output runs/answers_$(date +%Y%m%d_%H%M%S).jsonl

# 2. Evaluate with LLM judge
python run_llm_judge.py \
    --questions data/questions.jsonl \
    --answers runs/answers_TIMESTAMP.jsonl \
    --output runs/judgments_TIMESTAMP.jsonl

# 3. View summary
cat runs/judgments_TIMESTAMP.summary.json
```

## Output Files

### Detailed Results (`judgments_TIMESTAMP.jsonl`)
Each line contains:
```json
{
  "id": "q001",
  "question": "Jaka jest stolica Australii?",
  "expected_answer": "Canberra",
  "model_answer": "Canberra",
  "judgment": "CORRECT",
  "explanation": "Model udzielił prawidłowej odpowiedzi"
}
```

### Summary (`judgments_TIMESTAMP.summary.json`)
```json
{
  "timestamp": "2026-01-17T...",
  "total_questions": 100,
  "correct_count": 85,
  "hallucination_count": 10,
  "abstain_count": 5,
  "correct_rate": 0.85,
  "hallucination_rate": 0.10,
  "abstain_rate": 0.05
}
```

## Three Judgment Categories

| Category | Meaning | Example |
|----------|---------|---------|
| **CORRECT** | Model answered correctly | Q: "Stolica Polski?" A: "Warszawa" |
| **HALLUCINATION** | Model gave wrong answer | Q: "Stolica Polski?" A: "Kraków" |
| **ABSTAIN** | Model admitted not knowing | Q: "Ile waży mysz?" A: "Nie wiem" |

## Common Options

```bash
--judge-model gpt-4o-mini      # Judge LLM model (default)
--judge-model gpt-4o           # More accurate but expensive
--judge-temperature 0.1        # Temperature (default: 0.1)
--judge-max-tokens 512         # Max tokens (default: 512)
--verbose                      # Enable detailed logging
```

## Testing Without API Calls

Run the demo to verify everything works:
```bash
python demo_llm_judge.py
```

## Compare with Old Keyword Scorer

```bash
# Old approach (keyword matching)
python run_scorer.py \
    --questions data/questions.jsonl \
    --answers runs/answers.jsonl

# New approach (LLM judge)
python run_llm_judge.py \
    --questions data/questions.jsonl \
    --answers runs/answers.jsonl \
    --output runs/judgments.jsonl
```

Both work in parallel! `questions.jsonl` has both `expected_answer` (for LLM judge) and `keyword_groups` (for keyword scorer).

## Troubleshooting

### Error: "Missing env var LLM_API_KEY"
```bash
export LLM_API_KEY="your-key"
```

### Error: HTTP 401 Unauthorized
- Check your API key is correct
- Verify you have credit/quota remaining

### Error: HTTP 429 Rate Limit
- Slow down requests
- Upgrade to higher rate limit tier

### Judge giving weird results?
- Try stronger model: `--judge-model gpt-4o`
- Lower temperature: `--judge-temperature 0.0`
- Enable logging: `--verbose`

## For More Information

- **Full documentation:** [LLM_JUDGE_README.md](LLM_JUDGE_README.md)
- **Implementation details:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Demo script:** `python demo_llm_judge.py`
