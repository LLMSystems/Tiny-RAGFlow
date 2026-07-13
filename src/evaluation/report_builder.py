# src/evaluation/report_builder.py

from typing import List, Dict


class BenchmarkReportBuilder:

    @staticmethod
    def build_markdown(results: List[Dict], sort_by: str) -> str:
        """
        Build a full markdown table including ALL metrics,
        using the nested summary format:

        {
          "name": "FAISS",
          "by_k": {
            3: { "HitRate": ..., "Recall": ..., "Precision": ..., "NDCG": ... },
            5: { ... },
            10: { ... }
          },
          "MRR": ...
        }
        """

        # extract sorted list of K values from first entry
        sample_entry = results[0]
        k_values = list(sample_entry["summary"]["by_k"].keys())
        k_values.sort()

        # build header
        md = []
        md.append("## Retriever Benchmark Results\n")
        md.append(f"Sorted by **{sort_by}**\n")

        header_cols = ["Rank", "Retriever", "K", "HitRate", "Recall", "Precision", "NDCG", "MRR"]
        md.append("| " + " | ".join(header_cols) + " |")
        md.append("|" + "|".join(["---"] * len(header_cols)) + "|")

        # build table rows
        for rank, entry in enumerate(results, start=1):
            name = entry["name"]
            mrr_val = entry["summary"]["MRR"]

            for k in k_values:
                metrics = entry["summary"]["by_k"][k]
                row = [
                    str(rank),
                    name,
                    str(k),
                    f"{metrics['HitRate']:.4f}",
                    f"{metrics['Recall']:.4f}",
                    f"{metrics['Precision']:.4f}",
                    f"{metrics['NDCG']:.4f}",
                    f"{mrr_val:.4f}"
                ]
                md.append("| " + " | ".join(row) + " |")

        return "\n".join(md)

    @staticmethod
    def build_console(results: List[Dict], sort_by: str) -> str:
        """
        Console-friendly summary printing only sort_by metric.
        """
        lines = []
        lines.append("\n===== Retriever Benchmark Results =====\n")
        header = f"{'Rank':<5} | {'Retriever':<12} | {sort_by:<12}"
        lines.append(header)
        lines.append("-" * len(header))

        for idx, entry in enumerate(results, start=1):
            name = entry["name"]
            score = entry["summary"]["by_k"][max(entry['summary']['by_k'].keys())]["NDCG"]
            lines.append(f"{idx:<5} | {name:<12} | {score:<12.4f}")

        return "\n".join(lines)

    @staticmethod
    def build_json(results: List[Dict], sort_by: str, elapsed_time: None) -> str:
        import json
        output = {
            "sorted_by": sort_by,
            "elapsed_time": elapsed_time,
            "results": results
        }
        return json.dumps(output, indent=2, ensure_ascii=False)
