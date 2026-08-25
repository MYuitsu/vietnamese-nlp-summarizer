"""Automated Benchmark Runner and Academic Latex/Markdown Table Generator."""

import time
import os
import sys
from typing import List, Dict
import pandas as pd
from tabulate import tabulate

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from data.benchmark_samples import BENCHMARK_SAMPLES
from nlp_summary.evaluation.rouge import RougeEvaluator
from nlp_summary.models.evaluation import BenchmarkReport, BenchmarkSummaryRow
from nlp_summary.models.summary import LengthConstraint
from nlp_summary.summarizer.engine import ExtractiveSummarizerEngine


class BenchmarkRunner:
    """Runs automated evaluations across benchmark samples and generates scientific reports."""

    ALGORITHMS = ["tfidf", "textrank", "yake", "keybert"]

    def __init__(self, samples: List[Dict] = None):
        self.samples = samples or BENCHMARK_SAMPLES
        self.engine = ExtractiveSummarizerEngine(language="vi")
        self.evaluator = RougeEvaluator()

    def run_benchmark(self) -> BenchmarkReport:
        """Executes sweep across algorithms and aggregates ROUGE and latency scores."""
        algo_scores: Dict[str, Dict[str, List[float]]] = {
            algo: {
                "rouge_1_f1": [],
                "rouge_2_f1": [],
                "rouge_l_f1": [],
                "keyword_retention": [],
                "latency_ms": [],
            }
            for algo in self.ALGORITHMS
        }

        for item in self.samples:
            doc_text = item["document"]
            ref_summary = item["reference_summary"]
            title = item.get("title", None)

            for algo in self.ALGORITHMS:
                start = time.time()
                sum_res = self.engine.summarize(
                    text=doc_text,
                    title=title,
                    algorithm=algo,
                    top_k_keywords=10,
                    length_constraint=LengthConstraint(mode="sentence_count", value=2),
                )
                latency = (time.time() - start) * 1000.0

                eval_res = self.evaluator.evaluate(sum_res.summary_text, ref_summary)

                algo_scores[algo]["rouge_1_f1"].append(eval_res.rouge_1.f1_score)
                algo_scores[algo]["rouge_2_f1"].append(eval_res.rouge_2.f1_score)
                algo_scores[algo]["rouge_l_f1"].append(eval_res.rouge_l.f1_score)
                algo_scores[algo]["keyword_retention"].append(eval_res.keyword_retention)
                algo_scores[algo]["latency_ms"].append(latency)

        # Aggregate averages
        summary_rows: List[BenchmarkSummaryRow] = []
        table_dict_list = []

        for algo in self.ALGORITHMS:
            r1_mean = sum(algo_scores[algo]["rouge_1_f1"]) / len(self.samples)
            r2_mean = sum(algo_scores[algo]["rouge_2_f1"]) / len(self.samples)
            rl_mean = sum(algo_scores[algo]["rouge_l_f1"]) / len(self.samples)
            ret_mean = sum(algo_scores[algo]["keyword_retention"]) / len(self.samples)
            lat_mean = sum(algo_scores[algo]["latency_ms"]) / len(self.samples)

            row = BenchmarkSummaryRow(
                algorithm=algo.upper(),
                rouge_1_f1=round(r1_mean, 4),
                rouge_2_f1=round(r2_mean, 4),
                rouge_l_f1=round(rl_mean, 4),
                keyword_retention=round(ret_mean, 4),
                avg_latency_ms=round(lat_mean, 2),
            )
            summary_rows.append(row)
            table_dict_list.append({
                "Algorithm": row.algorithm,
                "ROUGE-1 F1": f"{row.rouge_1_f1:.4f}",
                "ROUGE-2 F1": f"{row.rouge_2_f1:.4f}",
                "ROUGE-L F1": f"{row.rouge_l_f1:.4f}",
                "KW Retention": f"{row.keyword_retention * 100:.1f}%",
                "Latency (ms)": f"{row.avg_latency_ms:.1f}",
            })

        df = pd.DataFrame(table_dict_list)
        md_table = tabulate(df, headers="keys", tablefmt="github", showindex=False)
        latex_table = tabulate(df, headers="keys", tablefmt="latex_booktabs", showindex=False)

        return BenchmarkReport(
            dataset_name="VietNews Sample Corpus",
            sample_count=len(self.samples),
            results=summary_rows,
            markdown_table=md_table,
            latex_table=latex_table,
        )


if __name__ == "__main__":
    runner = BenchmarkRunner()
    report = runner.run_benchmark()
    print("\n=== BENCHMARK REPORT (MARKDOWN) ===\n")
    print(report.markdown_table)
    print("\n=== LATEX TABLE FOR ACADEMIC PAPER (SN-JNL / LLNCS / CTU) ===\n")
    print(report.latex_table)
