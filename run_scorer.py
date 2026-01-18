from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.dataset import load_questions, load_answers
from src.scorer import score_answer

def main():
    ap = argparse.ArgumentParser(description="Stage-1 scorer: keyword-groups (OR/AND) presence check.")
    ap.add_argument("--questions", type=str, required=True, help="Path to questions JSONL")
    ap.add_argument("--answers", type=str, required=True, help="Path to answers JSONL (id + answer)")
    ap.add_argument("--no-diacritics-fold", action="store_true", help="Disable diacritics folding (ą!=a etc.)")
    ap.add_argument("--report", type=str, default="", help="Optional path to save per-item report JSONL")
    args = ap.parse_args()

    items = load_questions(args.questions)
    ans = load_answers(args.answers)

    total = 0
    correct = 0
    missing_answers = 0

    report_lines = []
    for it in items:
        total += 1
        a = ans.get(it.id, "")
        if not a:
            missing_answers += 1
        res = score_answer(a, it.keyword_groups, fold_diacritics=not args.no_diacritics_fold)
        correct += int(res.correct)

        report_lines.append({
            "id": it.id,
            "question": it.question,
            "keyword_groups": it.keyword_groups,
            "answer": a,
            "correct": res.correct,
            "matched_groups": [
                {"group": gm.group, "matched_variant": gm.matched_variant}
                for gm in res.matched_groups
            ],
            "missing_groups": res.missing_groups
        })

    acc = correct / total if total else 0.0

    print(f"Questions: {total}")
    print(f"Answers missing/empty: {missing_answers}")
    print(f"Accuracy (all groups satisfied): {acc:.3f} ({correct}/{total})")

    if args.report:
        outp = Path(args.report)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for row in report_lines:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved report to: {outp}")

if __name__ == "__main__":
    main()
