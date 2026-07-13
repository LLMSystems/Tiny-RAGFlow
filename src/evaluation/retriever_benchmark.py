from typing import List, Dict, Any
import os
import time
from datetime import datetime
import logging
from .dataset_loader import EvaluationDataset, MultiIntentEvaluationDataset
from .evaluator import RetrieverEvaluator, MultiIntentRetrieverEvaluator
from .report_builder import BenchmarkReportBuilder


class RetrieverBenchmark:
    def __init__(
        self,
        retrievers: List[tuple],
        eval_dataset: Any,
        type: str = "single"
    ):
        self.retrievers = retrievers
        self.eval_dataset = eval_dataset
        self.type = type
        self.logger = self._setup_logger()
        self.logger.info(f"Initialized RetrieverBenchmark with {len(retrievers)} retrievers on {type} intent dataset.")
        self.eval_function = MultiIntentRetrieverEvaluator if type == "multi" else RetrieverEvaluator


    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    async def run(
        self,
        top_k: List[int] = [5],
        sort_by: tuple = None,
        batch_size: int = 8,
        task: str = None,
        output_dir: str = "output"
    ) -> List[Dict[str, Any]]:
        self.logger.info("Starting Retriever Benchmark...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results_list = []
        time_elapsed = {}

        for name, retriever, task in self.retrievers:
            evaluator = self.eval_function(
                retriever,
                self.eval_dataset,
                task=task
            )

            start_time = time.time()
            results = await evaluator.evaluate(top_k=top_k, batch_size=batch_size, task=task)
            end_time = time.time()
            elapsed = end_time - start_time
            time_elapsed[name] = elapsed
            
            metrics = results["summary"]
            details = results["details"]

            result_entry = {
                "name": name,
                "summary": metrics,
                "details": details
            }
            results_list.append(result_entry)
            
        # default sort metric = NDCG@max_k
        if sort_by is None:
            max_k = max(top_k)
            sort_metric = "NDCG"
            sort_k = max_k
            sort_by = (sort_metric, sort_k)
        else:
            sort_metric, sort_k = sort_by
            
        if sort_metric == "MRR":
            sort_k = None  # MRR is not dependent on K

        # sorting using nested summary structure
        results_list.sort(
            key=lambda entry: entry["summary"]["by_k"][sort_k][sort_metric] if sort_k is not None else entry["summary"][sort_metric],
            reverse=True
        )
        
        sort_by_str = f"{sort_metric}@{sort_k}" if sort_k is not None else sort_metric
        
        report_console = BenchmarkReportBuilder.build_console(results_list, sort_by_str)
        report_markdown = BenchmarkReportBuilder.build_markdown(results_list, sort_by_str)
        report_json = BenchmarkReportBuilder.build_json(results_list, sort_by_str, time_elapsed)
        
        # save reports
        file_name = f"retriever_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.type}_intent"
        with open(os.path.join(output_dir, f"{file_name}.md"), "w") as f:
            f.write(report_markdown)
            self.logger.info(f"Markdown report saved to {file_name}.md")
        with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
            f.write(report_json)
            self.logger.info(f"JSON report saved to {file_name}.json")
        
        self.logger.info("Retriever Benchmark Completed.")
        
        return {
            "results": results_list,
            "sort_by": sort_by,
            "console": report_console,
            "markdown": report_markdown,
            "json": report_json
        }