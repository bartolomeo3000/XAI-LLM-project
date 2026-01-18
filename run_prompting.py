from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from src.dataset import load_questions, load_answers
from src.prompts import PROMPT_VARIANTS
from src.llm_clients import LLMConfig, make_client, MockClient
from src.scorer import score_answer
from src.metrics import is_abstain, summarize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_temperatures(arg: str) -> list[float]:
    vals: list[float] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No temperatures parsed")
    return vals

def main():
    ap = argparse.ArgumentParser(description="Stage-2: prompt the model with 3 prompt variants and score outputs.")
    ap.add_argument("--questions", type=str, required=True, help="Questions JSONL (id, question, keyword_groups)")
    ap.add_argument("--provider", type=str, default="openai_compat", choices=["openai_compat","mock"])
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name (provider-specific)")

    # Backward-compatible single temperature:
    ap.add_argument("--temperature", type=float, default=0.2, help="Single temperature (legacy). Ignored if --temperatures is set.")
    # New: multiple temperatures for comparison
    ap.add_argument("--temperatures", type=str, default="", help="Comma-separated temperatures, e.g. '0,0.7'")

    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--outdir", type=str, default="runs", help="Output directory for run files")
    ap.add_argument("--mock-answers", type=str, default="", help="For provider=mock: answers JSONL with id+answer")

    args = ap.parse_args()
    
    logger.info("Starting run_prompting.py")
    logger.info(f"Arguments: provider={args.provider}, model={args.model}, questions={args.questions}")

    items = load_questions(args.questions)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"answers_{run_id}.jsonl"
    summary_path = outdir / f"summary_{run_id}.json"
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {outdir}")

    temps = parse_temperatures(args.temperatures) if args.temperatures.strip() else [float(args.temperature)]
    logger.info(f"Using temperatures: {temps}")

    mock_answers_by_id = None
    if args.provider == "mock":
        if not args.mock_answers:
            raise SystemExit("--mock-answers is required when --provider mock")
        logger.info(f"Loading mock answers from {args.mock_answers}")
        mock_answers_by_id = load_answers(args.mock_answers)
        logger.info(f"Loaded {len(mock_answers_by_id)} mock answers")

    # Prompting loop
    logger.info(f"Starting prompting loop: {len(temps)} temperature(s), {len(items)} questions, {len(PROMPT_VARIANTS)} prompt variants")
    total_calls = len(temps) * len(items) * len(PROMPT_VARIANTS)
    logger.info(f"Total API calls to make: {total_calls}")
    
    call_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for temp_idx, temp in enumerate(temps, 1):
            logger.info(f"Processing temperature {temp_idx}/{len(temps)}: {temp}")
            cfg = LLMConfig(
                provider=args.provider,
                model=args.model,
                temperature=float(temp),
                max_tokens=args.max_tokens,
            )
            client = make_client(cfg, mock_answers_by_id=mock_answers_by_id)

            for item_idx, it in enumerate(items, 1):
                logger.debug(f"Processing question {item_idx}/{len(items)}: id={it.id}")
                for pv_idx, pv in enumerate(PROMPT_VARIANTS, 1):
                    call_count += 1
                    logger.debug(f"Using prompt variant {pv_idx}/{len(PROMPT_VARIANTS)}: {pv.name} (call {call_count}/{total_calls})")
                    user_prompt = pv.template.format(question=it.question)

                    if isinstance(client, MockClient):
                        client.set_current_id(it.id)

                    logger.debug(f"Generating answer for question {it.id} with {pv.name}")
                    answer = client.generate(system=pv.system, user=user_prompt)
                    logger.debug(f"Received answer: {answer[:50]}..." if len(answer) > 50 else f"Received answer: {answer}")

                    row = {
                        "run_id": run_id,
                        "ts_utc": now_iso(),
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "temperature": cfg.temperature,
                        "max_tokens": cfg.max_tokens,
                        "prompt_variant": pv.name,
                        "id": it.id,
                        "question": it.question,
                        "user_prompt": user_prompt,
                        "answer": answer,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logger.info(f"Completed {call_count} API calls")
    logger.info(f"Answers saved to {out_path}")

    # Score outputs (always from saved file)
    logger.info("Starting scoring phase")
    # Structure: summary[temperature][prompt_variant] = metrics
    by_temp_variant: dict[str, dict[str, list[tuple[bool,bool]]]] = {
        str(t): {pv.name: [] for pv in PROMPT_VARIANTS} for t in temps
    }

    id_to_item = {it.id: it for it in items}

    scored_count = 0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            temp_key = str(obj["temperature"])
            v = obj["prompt_variant"]
            qid = obj["id"]
            answer = obj["answer"]

            it = id_to_item[qid]
            res = score_answer(answer, it.keyword_groups, fold_diacritics=True)
            abst = is_abstain(answer)
            by_temp_variant[temp_key][v].append((res.correct, abst))
            scored_count += 1
    
    logger.info(f"Scored {scored_count} answers")

    logger.info("Computing summary statistics")
    summary = {}
    for temp_key, per_variant in by_temp_variant.items():
        summary[temp_key] = {}
        for v, outcomes in per_variant.items():
            s = summarize(outcomes)
            summary[temp_key][v] = {
                "n": s.n,
                "accuracy": s.accuracy,
                "abstain_rate": s.abstain_rate,
                "incorrect_rate": s.incorrect_rate,
                "hallucination_rate": s.hallucination_rate,
            }
            logger.debug(f"Temperature={temp_key}, Variant={v}: acc={s.accuracy:.3f}, abstain={s.abstain_rate:.3f}, halluc={s.hallucination_rate:.3f}")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "questions_path": str(Path(args.questions).resolve()),
                "answers_path": str(out_path.resolve()),
                "created_utc": now_iso(),
                "config": {
                    "provider": args.provider,
                    "model": args.model,
                    "temperatures": temps,
                    "max_tokens": args.max_tokens,
                },
                "summary_by_temperature_then_prompt": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    logger.info(f"Summary saved to {summary_path}")

    print(f"Saved answers: {out_path}")
    print(f"Saved summary: {summary_path}")
    print("\nSummary by temperature and prompt variant:")
    for temp_key in summary:
        print(f"Temperature={temp_key}")
        for pv in [p.name for p in PROMPT_VARIANTS]:
            s = summary[temp_key][pv]
            print(
                f"  - {pv}: acc={s['accuracy']:.3f}, abstain={s['abstain_rate']:.3f}, "
                f"halluc={s['hallucination_rate']:.3f} (n={s['n']})"
            )

if __name__ == "__main__":
    main()
