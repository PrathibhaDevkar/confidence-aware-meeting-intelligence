"""Runs ROUGE-L, BERTScore, and LLM-judge across the gold meetings' LLM-arm
summaries, and appends the results to reports/evaluation_summary.md
(action_item_prf.py's output, run first). Both are secondary/supporting
signals - see eval/metrics_rouge_bertscore.py and eval/llm_judge.py
docstrings for why neither is the headline result.

Usage: python3 eval/run_summary_metrics.py
(run after eval/action_item_prf.py has written the report's first section)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.llm_judge import judge_summary
from eval.metrics_rouge_bertscore import bertscore_batch, rouge_l
from pipeline.schema import SummaryOutput, Transcript

GOLD_IDS_PATH = Path("data/eval_gold/selected_meetings.txt")
LLM_DIR = Path("data/outputs/llm")
PROCESSED_ROOT = Path("data/processed")
REPORT_OUT = Path("reports/evaluation_summary.md")


def main() -> None:
    meeting_ids = [m.strip() for m in GOLD_IDS_PATH.read_text().splitlines() if m.strip()]

    references, hypotheses, rouge_scores, judge_scores, valid_ids = [], [], [], [], []
    for i, meeting_id in enumerate(meeting_ids, 1):
        dataset = meeting_id.split("_", 1)[0]
        transcript = Transcript.model_validate_json(
            (PROCESSED_ROOT / dataset / f"{meeting_id}.json").read_text()
        )
        llm_path = LLM_DIR / f"{meeting_id}.json"
        if not llm_path.exists() or not transcript.reference_summary:
            continue

        summary = SummaryOutput.model_validate_json(llm_path.read_text())
        print(f"[{i}/{len(meeting_ids)}] {meeting_id}")

        references.append(transcript.reference_summary)
        hypotheses.append(summary.abstractive_summary)
        rouge_scores.append(rouge_l(transcript.reference_summary, summary.abstractive_summary))
        valid_ids.append(meeting_id)

        judged = judge_summary(transcript, summary.abstractive_summary)
        judge_scores.append(judged)
        print(f"  rouge-L={rouge_scores[-1]:.3f}  judge: coherence={judged['coherence']} "
              f"consistency={judged['consistency']} fluency={judged['fluency']} relevance={judged['relevance']}")

    print("\nComputing BERTScore (batch)...")
    bert_f1s = bertscore_batch(references, hypotheses)

    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_bert = sum(bert_f1s) / len(bert_f1s)
    avg_judge = {
        dim: sum(j[dim] for j in judge_scores) / len(judge_scores)
        for dim in ["coherence", "consistency", "fluency", "relevance"]
    }

    lines = ["\n## Summarization quality (secondary signals - see module docstrings)\n"]
    lines.append(f"n = {len(valid_ids)} gold meetings (LLM arm)\n")
    lines.append("| Metric | Average |")
    lines.append("|---|---|")
    lines.append(f"| ROUGE-L (cheap regression check only) | {avg_rouge:.3f} |")
    lines.append(f"| BERTScore F1 (distilbert, cheap regression check only) | {avg_bert:.3f} |")
    for dim, val in avg_judge.items():
        lines.append(f"| LLM-judge {dim} (1-5, same model self-judging - weak signal) | {val:.2f} |")

    report_section = "\n".join(lines) + "\n"
    with REPORT_OUT.open("a") as f:
        f.write(report_section)

    print(report_section)
    print(f"Appended to {REPORT_OUT}")


if __name__ == "__main__":
    main()
