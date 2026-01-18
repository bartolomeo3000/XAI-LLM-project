#!/usr/bin/env python3
"""
LLM Judge Scorer - Evaluate model answers using an LLM judge.

This script uses an LLM to judge whether model answers are:
1. CORRECT - answered correctly
2. HALLUCINATION - answered incorrectly with made-up information
3. ABSTAIN - admitted not knowing (no hallucination)

Usage:
    python run_llm_judge.py --questions data/questions.jsonl --answers runs/answers_XXX.jsonl --output runs/judgments_XXX.jsonl

Environment variables:
    LLM_API_KEY: API key for the judge LLM
    LLM_BASE_URL: Base URL for the judge LLM API (default: https://api.openai.com/v1)
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from src.dataset import load_questions, load_answers_with_variants
from src.llm_clients import LLMConfig, make_client
from src.llm_judge import judge_answer, summarize_judgments, JudgmentCategory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate model answers using an LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("--questions", type=str, required=True, help="Path to questions JSONL with expected_answer field")
    ap.add_argument("--answers", type=str, required=True, help="Path to answers JSONL (id + answer)")
    ap.add_argument("--output", type=str, required=True, help="Path to save judgment results JSONL")
    
    # Judge LLM configuration
    ap.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="Judge model name (default: gpt-4o-mini)")
    ap.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature (default: 0.0)")
    ap.add_argument("--judge-max-tokens", type=int, default=512, help="Judge max tokens (default: 512)")
    ap.add_argument("--judge-timeout", type=int, default=60, help="Judge API timeout in seconds (default: 60)")
    
    ap.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    
    args = ap.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load questions and answers
    logger.info(f"Loading questions from {args.questions}")
    questions = load_questions(args.questions)
    
    logger.info(f"Loading answers from {args.answers}")
    answer_records = load_answers_with_variants(args.answers)
    
    # Group answers by question id and variant
    from collections import defaultdict
    answers_by_id_variant = defaultdict(dict)
    for record in answer_records:
        qid = str(record["id"])
        variant = record.get("prompt_variant", "unknown")
        answers_by_id_variant[qid][variant] = record
    
    logger.info(f"Loaded {len(answer_records)} answer records for {len(answers_by_id_variant)} questions")
    variants = set()
    for ans_dict in answers_by_id_variant.values():
        variants.update(ans_dict.keys())
    logger.info(f"Prompt variants found: {sorted(variants)}")
    
    # Check that all questions have expected_answer
    missing_expected = [q for q in questions if not q.expected_answer]
    if missing_expected:
        logger.warning(f"{len(missing_expected)} questions are missing expected_answer field")
        logger.warning(f"First few: {[q.id for q in missing_expected[:5]]}")
    
    # Initialize judge LLM
    judge_cfg = LLMConfig(
        provider="openai_compat",
        model=args.judge_model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        timeout_s=args.judge_timeout
    )
    logger.info(f"Initializing judge LLM: {judge_cfg.model}")
    judge_client = make_client(judge_cfg)
    
    # Process each question and variant
    results = []
    missing_answers = 0
    total_evaluations = 0
    
    # Get all unique variants
    all_variants = set()
    for ans_dict in answers_by_id_variant.values():
        all_variants.update(ans_dict.keys())
    all_variants = sorted(all_variants)
    
    logger.info(f"Processing {len(questions)} questions with {len(all_variants)} variants each...")
    logger.info(f"Total evaluations: {len(questions) * len(all_variants)}")
    
    for i, q in enumerate(questions, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(questions)} questions")
        
        variants_for_q = answers_by_id_variant.get(q.id, {})
        
        # Process each variant for this question
        for variant in all_variants:
            total_evaluations += 1
            
            if variant not in variants_for_q:
                missing_answers += 1
                logger.debug(f"No answer for {q.id} variant '{variant}'")
                model_answer = ""
                answer_record = {}
            else:
                answer_record = variants_for_q[variant]
                model_answer = str(answer_record.get("answer", ""))
            
            # Get judgment
            judgment = judge_answer(
                question=q.question,
                model_answer=model_answer,
                expected_answer=q.expected_answer,
                judge_client=judge_client
            )
            
            # Store result with full context
            result = {
                "id": q.id,
                "prompt_variant": variant,
                "question": q.question,
                "expected_answer": q.expected_answer,
                "model_answer": model_answer,
                "judgment": judgment.category.value,
                "explanation": judgment.explanation,
                "model": answer_record.get("model", ""),
                "temperature": answer_record.get("temperature", ""),
                "user_prompt": answer_record.get("user_prompt", "")
            }
            results.append(result)
    
    # Calculate summary overall and per variant
    from src.llm_judge import JudgmentResult
    
    # Overall summary
    judgment_objs = [
        JudgmentResult(category=JudgmentCategory(r["judgment"]), explanation=r["explanation"])
        for r in results
    ]
    summary = summarize_judgments(judgment_objs)
    
    # Per-variant summaries
    variant_summaries = {}
    for variant in all_variants:
        variant_results = [r for r in results if r["prompt_variant"] == variant]
        variant_judgments = [
            JudgmentResult(category=JudgmentCategory(r["judgment"]), explanation=r["explanation"])
            for r in variant_results
        ]
        variant_summaries[variant] = summarize_judgments(variant_judgments)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Save summary
    summary_path = output_path.with_suffix(".summary.json")
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "questions_file": args.questions,
        "answers_file": args.answers,
        "judge_model": args.judge_model,
        "total_questions": len(questions),
        "total_evaluations": total_evaluations,
        "prompt_variants": all_variants,
        "missing_answers": missing_answers,
        "overall": {
            "correct_count": int(summary.correct_rate * summary.n),
            "hallucination_count": int(summary.hallucination_rate * summary.n),
            "abstain_count": int(summary.abstain_rate * summary.n),
            "correct_rate": summary.correct_rate,
            "hallucination_rate": summary.hallucination_rate,
            "abstain_rate": summary.abstain_rate
        },
        "by_variant": {}
    }
    
    # Add per-variant statistics
    for variant, var_summary in variant_summaries.items():
        summary_data["by_variant"][variant] = {
            "total": var_summary.n,
            "correct_count": int(var_summary.correct_rate * var_summary.n),
            "hallucination_count": int(var_summary.hallucination_rate * var_summary.n),
            "abstain_count": int(var_summary.abstain_rate * var_summary.n),
            "correct_rate": var_summary.correct_rate,
            "hallucination_rate": var_summary.hallucination_rate,
            "abstain_rate": var_summary.abstain_rate
        }
    
    logger.info(f"Saving summary to {summary_path}")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total questions: {len(questions)}")
    print(f"Prompt variants: {', '.join(all_variants)}")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Missing answers: {missing_answers}")
    
    print(f"\nOVERALL RESULTS (all variants combined):")
    print(f"  Correct:        {summary.correct_rate:6.1%} ({int(summary.correct_rate * summary.n)}/{summary.n})")
    print(f"  Hallucination:  {summary.hallucination_rate:6.1%} ({int(summary.hallucination_rate * summary.n)}/{summary.n})")
    print(f"  Abstain:        {summary.abstain_rate:6.1%} ({int(summary.abstain_rate * summary.n)}/{summary.n})")
    
    print(f"\nRESULTS BY PROMPT VARIANT:")
    for variant in all_variants:
        var_summary = variant_summaries[variant]
        print(f"\n  {variant.upper()}:")
        print(f"    Correct:        {var_summary.correct_rate:6.1%} ({int(var_summary.correct_rate * var_summary.n)}/{var_summary.n})")
        print(f"    Hallucination:  {var_summary.hallucination_rate:6.1%} ({int(var_summary.hallucination_rate * var_summary.n)}/{var_summary.n})")
        print(f"    Abstain:        {var_summary.abstain_rate:6.1%} ({int(var_summary.abstain_rate * var_summary.n)}/{var_summary.n})")
    
    print("\n" + "="*70)
    print(f"Detailed results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()
