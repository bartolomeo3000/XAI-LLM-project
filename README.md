# Hallucination analysis (Stage 1): dataset + keyword-group scorer

This stage contains:
- A minimal dataset format: `question` + `keyword_groups` (OR within a group, AND across groups).
- A robust keyword scorer with normalization (case, punctuation, whitespace, optional diacritics folding).
- A small sample dataset + sample "model answers" file to test the scorer.

## Dataset format (JSONL)

Each line is a JSON object:

### Preferred format: keyword_groups (OR/AND)
```json
{"id":"q001","question":"...","keyword_groups":[["Warszawa","Warsaw"]]}
```

Interpretation:
- **OR within a group**: at least one variant in the same group must appear in the answer
- **AND across groups**: every group must be satisfied

Example (two required pieces of info):
```json
{"id":"q100","question":"...","keyword_groups":[["Jan"],["Kowalski"]]}
```

### Backward-compatible format: keywords (AND)
```json
{"id":"q002","question":"...","keywords":["Jan","Kowalski"]}
```
This is treated as:
```json
"keyword_groups":[["Jan"],["Kowalski"]]
```

## Quick start
From the project folder:

Score sample answers:
```bash
python run_scorer.py --questions data/questions_sample.jsonl --answers data/answers_sample.jsonl
```

Save a per-item report:
```bash
python run_scorer.py --questions data/questions_sample.jsonl --answers data/answers_sample.jsonl --report runs/report.jsonl
```


## Stage 2: Prompting with 3 variants + hallucination rate

### Using an OpenAI-compatible API (recommended)
Set:
- `LLM_API_KEY` (required)
- `LLM_BASE_URL` (optional; default `https://api.openai.com/v1`)

Run:
```bash
python run_prompting.py --questions data/questions.jsonl --provider openai_compat --model gpt-4o-mini --temperature 0.2
```

This will create:
- `runs/answers_<timestamp>.jsonl`
- `runs/summary_<timestamp>.json`

### Using mock answers (offline test)
```bash
python run_prompting.py --questions data/questions_sample.jsonl --provider mock --mock-answers data/answers_sample.jsonl
```


### Comparing temperatures (recommended: 0 and 0.7)
Run:
```bash
python run_prompting.py --questions data/questions.jsonl --provider openai_compat --model gpt-4o-mini --temperatures 0,0.7
```

This produces a summary nested by:
`temperature -> prompt_variant`.


## Logging

The project includes comprehensive logging to help track what's happening during execution:

### Log Levels
- **INFO**: High-level progress (questions loaded, API calls completed, scoring progress)
- **DEBUG**: Detailed execution (individual API calls, answer scoring, keyword matching)
- **ERROR**: Errors and failures

### Default Configuration
By default, `run_prompting.py` logs at INFO level with the format:
```
YYYY-MM-DD HH:MM:SS - module_name - LEVEL - message
```

### Enabling Debug Logging
To see more detailed logs including individual API calls and scoring details, you can modify the logging level:

**Option 1: Edit run_prompting.py**
Change line 16 from `level=logging.INFO` to `level=logging.DEBUG`

**Option 2: Set environment variable**
```bash
$env:PYTHONVERBOSE = "1"
```

### Example Log Output
```
2026-01-12 20:30:45 - __main__ - INFO - Starting run_prompting.py
2026-01-12 20:30:45 - __main__ - INFO - Arguments: provider=openai_compat, model=gpt-4o-mini, questions=data/questions.jsonl
2026-01-12 20:30:45 - src.dataset - INFO - Loading questions from data/questions.jsonl
2026-01-12 20:30:45 - src.dataset - INFO - Successfully loaded 92 questions
2026-01-12 20:30:45 - __main__ - INFO - Run ID: 20260112_203045
2026-01-12 20:30:45 - __main__ - INFO - Using temperatures: [0.2]
2026-01-12 20:30:45 - src.llm_clients - INFO - Initialized OpenAICompatClient: model=gpt-4o-mini, temperature=0.2, max_tokens=256
2026-01-12 20:30:45 - __main__ - INFO - Starting prompting loop: 1 temperature(s), 92 questions, 3 prompt variants
2026-01-12 20:30:45 - __main__ - INFO - Total API calls to make: 276
2026-01-12 20:30:45 - __main__ - INFO - Processing temperature 1/1: 0.2
...
2026-01-12 20:35:12 - __main__ - INFO - Completed 276 API calls
2026-01-12 20:35:12 - __main__ - INFO - Answers saved to runs/answers_20260112_203045.jsonl
2026-01-12 20:35:12 - __main__ - INFO - Starting scoring phase
2026-01-12 20:35:12 - __main__ - INFO - Scored 276 answers
2026-01-12 20:35:12 - __main__ - INFO - Computing summary statistics
2026-01-12 20:35:12 - __main__ - INFO - Summary saved to runs/summary_20260112_203045.json
```

## Provided dataset
A starter dataset is included at `data/questions.jsonl` (92 questions).
