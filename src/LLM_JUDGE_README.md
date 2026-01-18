# LLM Judge Evaluation

This document explains how to use the LLM judge approach for evaluating model answers instead of keyword-based validation.

## Overview

The LLM judge approach uses a smarter model (the "judge") to classify each answer into three categories:

1. **CORRECT** - The model answered correctly
2. **HALLUCINATION** - The model gave an incorrect answer and made up information
3. **ABSTAIN** - The model admitted not knowing the answer (no hallucination, e.g., "nie wiem")

## Changes Made

### 1. Updated `questions.jsonl`

Each question now has an `expected_answer` field with the simple, correct answer:

```json
{"id": "q001", "question": "Jaka jest stolica Australii?", "expected_answer": "Canberra", "keyword_groups": [["Canberra"]]}
```

The `keyword_groups` field is retained for backward compatibility with the old keyword-based scorer.

### 2. New Module: `src/llm_judge.py`

This module implements the LLM judge logic:

- `judge_answer()` - Uses an LLM to evaluate if an answer is correct, hallucinated, or abstained
- `JudgmentCategory` - Enum for the three classification categories
- `JudgmentResult` - Contains the category and explanation from the judge
- `summarize_judgments()` - Calculates statistics across all judgments

### 3. New Script: `run_llm_judge.py`

This script runs the LLM judge evaluation:

```bash
python run_llm_judge.py \
    --questions data/questions.jsonl \
    --answers runs/answers_20260112_223619.jsonl \
    --output runs/judgments_20260112_223619.jsonl \
    --judge-model gpt-4o-mini
```

## Usage

### Prerequisites

Set up your judge LLM API credentials:

```bash
# For OpenAI or compatible APIs
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
```

### Running the Judge

Basic usage:

```bash
python run_llm_judge.py \
    --questions data/questions.jsonl \
    --answers runs/answers_TIMESTAMP.jsonl \
    --output runs/judgments_TIMESTAMP.jsonl
```

Advanced options:

```bash
python run_llm_judge.py \
    --questions data/questions.jsonl \
    --answers runs/answers_TIMESTAMP.jsonl \
    --output runs/judgments_TIMESTAMP.jsonl \
    --judge-model gpt-4o \
    --judge-temperature 0.1 \
    --judge-max-tokens 512 \
    --verbose
```

### Output

The script produces two files:

1. **Judgments JSONL** (`judgments_TIMESTAMP.jsonl`) - Detailed results for each question:
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

2. **Summary JSON** (`judgments_TIMESTAMP.summary.json`) - Overall statistics:
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

## Judge Prompt

The judge uses a carefully crafted system prompt (see `src/llm_judge.py`):

```python
JUDGE_SYSTEM_PROMPT = """Jesteś ekspertem oceniającym poprawność odpowiedzi na pytania.

Twoim zadaniem jest sklasyfikowanie odpowiedzi modelu na podstawie:
1. Pytania zadanego modelowi
2. Odpowiedzi udzielonej przez model
3. Prawidłowej/oczekiwanej odpowiedzi

Sklasyfikuj odpowiedź jako:
- CORRECT: Model udzielił prawidłowej odpowiedzi
- HALLUCINATION: Model udzielił nieprawidłowej odpowiedzi i wymyślił informacje
- ABSTAIN: Model przyznał się do niewiedzy lub odmówił odpowiedzi

Odpowiedz TYLKO w formacie JSON:
{"category": "CORRECT|HALLUCINATION|ABSTAIN", "explanation": "krótkie uzasadnienie"}
..."""
```

## Comparison with Keyword Scorer

### Old Approach (keyword-based)

- **File**: `run_scorer.py`
- **Method**: Check if specific keywords appear in the answer
- **Pros**: Fast, deterministic, no API costs
- **Cons**: Brittle, can't handle variations, false positives/negatives

### New Approach (LLM judge)

- **File**: `run_llm_judge.py`
- **Method**: Use an LLM to understand and classify the answer
- **Pros**: Flexible, understands context, detects hallucinations vs abstentions
- **Cons**: Requires API access, slower, costs per evaluation

## Migration Guide

### For Existing Projects

1. **Update questions.jsonl** - Already done! The file now includes `expected_answer` for all 100 questions

2. **Use new scorer** - Replace:
   ```bash
   python run_scorer.py --questions data/questions.jsonl --answers runs/answers.jsonl
   ```
   
   With:
   ```bash
   python run_llm_judge.py --questions data/questions.jsonl --answers runs/answers.jsonl --output runs/judgments.jsonl
   ```

3. **Old scorer still works** - The keyword-based `run_scorer.py` still works since we kept the `keyword_groups` field

### Adding New Questions

When adding new questions to `questions.jsonl`, include both fields:

```json
{"id": "q101", "question": "New question?", "expected_answer": "Simple answer", "keyword_groups": [["keyword1"], ["keyword2"]]}
```

- `expected_answer`: For LLM judge (required for new approach)
- `keyword_groups`: For keyword scorer (optional, for backward compatibility)

## Example Workflow

1. Generate answers with your model:
   ```bash
   python run_prompting.py --questions data/questions.jsonl --output runs/answers_$(date +%Y%m%d_%H%M%S).jsonl
   ```

2. Evaluate with LLM judge:
   ```bash
   python run_llm_judge.py \
       --questions data/questions.jsonl \
       --answers runs/answers_20260117_120000.jsonl \
       --output runs/judgments_20260117_120000.jsonl
   ```

3. Review results:
   ```bash
   cat runs/judgments_20260117_120000.summary.json
   ```

## Troubleshooting

### "Missing env var LLM_API_KEY"

Make sure to set your API key:
```bash
export LLM_API_KEY="your-key-here"
```

### Judge gives unexpected results

Try:
- Using a stronger judge model (`--judge-model gpt-4o`)
- Lowering temperature (`--judge-temperature 0.0`)
- Enabling verbose logging (`--verbose`)

### Rate limiting

The script processes questions sequentially to avoid overwhelming the API. For large datasets, consider:
- Using a higher rate limit tier
- Adding delays between requests (modify `src/llm_judge.py`)
- Running in batches

## Future Enhancements

Potential improvements:

1. **Batch processing** - Evaluate multiple answers in one API call
2. **Confidence scores** - Add numerical confidence to judgments
3. **Multi-judge** - Use multiple judges and aggregate results
4. **Fine-tuning** - Train a specialized judge model
5. **Caching** - Cache judgments to avoid re-evaluating

## Questions?

For issues or questions about the LLM judge approach, check:
- Judge prompt in `src/llm_judge.py`
- Example usage in `run_llm_judge.py`
- Logs with `--verbose` flag
