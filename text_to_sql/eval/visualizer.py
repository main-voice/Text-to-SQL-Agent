"""This module contains classes to analyze and visualize the evaluation results."""

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from text_to_sql.eval.models import AnalyzerItem, BoxPlotItem, EvalResultItem
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class EvalResultAnalyzer:
    """Analyze evaluation results and visualize the results."""

    def __init__(self, eval_dir: str):
        self.eval_dir = Path(__file__).parent / eval_dir

    def analyze(self) -> List[AnalyzerItem]:
        """Analyze evaluation results and return the results."""
        results: List[AnalyzerItem] = []
        for platform_dir in self.eval_dir.iterdir():
            platform = platform_dir.name
            for eval_file in platform_dir.glob("prod_*_*.json"):
                # golden name: prod_{method}_{model}.json
                parts = eval_file.stem.split("_")
                if len(parts) != 3:
                    logger.warning(f"Invalid file name: {eval_file.stem}")
                    continue

                method, model_name = parts[1], parts[2]

                try:
                    with open(eval_file, "r", encoding="utf-8") as f:
                        eval_results = [EvalResultItem.parse_obj(data) for data in json.load(f)]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to load {eval_file}: {e}")
                    continue

                accuracy = sum(1 for r in eval_results if r.is_correct) / len(eval_results)

                # loading token usage and evaluation duration data
                token_usage = [r.token_usage for r in eval_results if r.token_usage is not None]
                eval_duration = [r.eval_duration for r in eval_results if r.eval_duration is not None]

                try:
                    token_usage_item = None
                    if token_usage:
                        token_usage_q1 = np.percentile(token_usage, 25)
                        token_usage_q3 = np.percentile(token_usage, 75)
                        iqr = token_usage_q3 - token_usage_q1
                        lower_bound = token_usage_q1 - 1.5 * iqr
                        upper_bound = token_usage_q3 + 1.5 * iqr
                        outliers = [t for t in token_usage if t < lower_bound or t > upper_bound]

                        token_usage_item = BoxPlotItem(
                            min_value=np.min(token_usage),
                            max_value=np.max(token_usage),
                            avg_value=np.mean(token_usage),
                            median=np.median(token_usage),
                            q1=token_usage_q1,
                            q3=token_usage_q3,
                            outliers=outliers,
                        )

                    eval_duration_item = None
                    if eval_duration:
                        eval_duration_q1 = np.percentile(eval_duration, 25)
                        eval_duration_q3 = np.percentile(eval_duration, 75)
                        iqr = eval_duration_q3 - eval_duration_q1
                        lower_bound = eval_duration_q1 - 1.5 * iqr
                        upper_bound = eval_duration_q3 + 1.5 * iqr
                        outliers = [t for t in eval_duration if t < lower_bound or t > upper_bound]

                        eval_duration_item = BoxPlotItem(
                            min_value=np.min(eval_duration),
                            max_value=np.max(eval_duration),
                            avg_value=np.mean(eval_duration),
                            median=np.median(eval_duration),
                            q1=eval_duration_q1,
                            q3=eval_duration_q3,
                            outliers=outliers,
                        )

                    analyzer_item = AnalyzerItem(
                        platform=platform,
                        llm_model=model_name,
                        method=method,
                        accuracy=accuracy,
                        token_usage=token_usage_item,
                        eval_duration=eval_duration_item,
                    )

                    results.append(analyzer_item)
                    logger.info(f"Added analyzer item for {model_name} using method {method} on {platform}")

                except Exception as e:
                    logger.error(f"Failed to calculate statistics for {eval_file}: {e}")

        return results


class EvalResultVisualizer:
    """Visualize the evaluation results."""

    def __init__(self) -> None:
        self.metrics_visualzer_func = {
            "accuracy": self.visualize_accuracy,
            "token_usage": self.visualize_token_usage,
            "eval_duration": self.visualize_eval_duration,
        }

    def visualize_accuracy(self, analyzer_items: List[AnalyzerItem], output_dir: str):
        """
        Visualize the accuracy of the evaluation results.
        """
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir

        plt.figure(figsize=(10, 6))
        bar_width = 0.2
        methods = sorted(set(item.method for item in analyzer_items))
        models = sorted(set(item.llm_model for item in analyzer_items))

        x = np.arange(len(methods))

        for i, model in enumerate(models):
            y = []
            for method in methods:
                logger.debug(f"Accuracy, model: {model}, method: {method}")
                item = next(
                    (item for item in analyzer_items if item.method == method and item.llm_model == model), None
                )
                accuracy = item.accuracy if item and item.accuracy else 0
                y.append(accuracy)

            plt.bar(x + i * bar_width, y, width=bar_width, label=model)

        plt.xticks(x + bar_width, methods)
        plt.xlabel("Methods")
        plt.ylabel("Accuracy")
        plt.title("Comparison of Accuracy")

        plt.legend()
        plt.tight_layout()

        output_file = output_dir.joinpath("accuracy.png")
        plt.savefig(output_file)
        plt.close()

    def visualize_token_usage(self, analyzer_items: List[AnalyzerItem], output_dir: str | Path):
        """
        Visualize the token usage of the evaluation results.
        """
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir

        plt.figure(figsize=(10, 6))
        methods = sorted(set(item.method for item in analyzer_items))
        models = sorted(set(item.llm_model for item in analyzer_items))

        data = []
        labels = []

        for model in models:
            for method in methods:
                logger.debug(f"Token usage, model: {model}, method: {method}")
                item = next(
                    (item for item in analyzer_items if item.method == method and item.llm_model == model), None
                )

                if item and item.token_usage:
                    values = [
                        item.token_usage.min_value,
                        item.token_usage.q1,
                        item.token_usage.median,
                        item.token_usage.q3,
                        item.token_usage.max_value,
                    ]
                    data.append(values)
                    labels.append(f"{model} & {method}")

        plt.boxplot(
            data,
            labels=labels,
            notch=False,
            showmeans=True,
            meanline=True,
            flierprops=dict(marker="o", markersize=5),
            medianprops=dict(linewidth=1.5, color="red"),
            meanprops=dict(linewidth=1.5, color="blue"),
        )

        plt.xlabel("Model & Eval Method")
        plt.ylabel("Token Usage")
        plt.yscale("log")
        plt.title("Comparison of Token Usage")
        plt.xticks(rotation=45)

        plt.tight_layout()

        output_file = output_dir.joinpath("token_usage_boxplot.png")
        plt.savefig(output_file)
        plt.close()

    def visualize_eval_duration(self, analyzer_items: List[AnalyzerItem], output_dir: str):
        """
        Visualize the evaluation duration of the evaluation results.
        """
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir

        plt.figure(figsize=(10, 6))
        methods = sorted(set(item.method for item in analyzer_items))
        models = sorted(set(item.llm_model for item in analyzer_items))

        data = []
        labels = []

        for model in models:
            for method in methods:
                logger.debug(f"Eval duration, model: {model}, method: {method}")
                item = next(
                    (item for item in analyzer_items if item.method == method and item.llm_model == model), None
                )

                if item and item.eval_duration:
                    values = [
                        item.eval_duration.min_value,
                        item.eval_duration.q1,
                        item.eval_duration.median,
                        item.eval_duration.q3,
                        item.eval_duration.max_value,
                    ]
                    data.append(values)
                    labels.append(f"{model} & {method}")

        plt.boxplot(
            data,
            labels=labels,
            notch=False,
            showmeans=True,
            meanline=True,
            flierprops=dict(marker="o", markersize=5),
            medianprops=dict(linewidth=1.5, color="red"),
            meanprops=dict(linewidth=1.5, color="blue"),
        )

        plt.xlabel("Model & Eval Method")

        plt.ylabel("Evaluation Duration (s)")
        plt.yscale("log")
        plt.title("Comparison of Evaluation Duration")
        plt.xticks(rotation=45)

        plt.tight_layout()

        output_file = output_dir.joinpath("eval_duration_boxplot.png")
        plt.savefig(output_file)
        plt.close()

    def visualize(self, analyzer_items: List[AnalyzerItem], output_dir: str, metrics: List[str] = None):
        """Visualize the evaluation results.

        Args:
            results (List[AnalyzerItem]): The evaluation results.
            output_dir (Path): The output directory to save the visualizations.
        """
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        supported_metrics = self.metrics_visualzer_func.keys()
        if metrics is None:
            metrics = supported_metrics
        else:
            missing_metrics = set(metrics) - supported_metrics
            if missing_metrics:
                logger.warning(f"Unsupported metrics: {missing_metrics}, skipping them.")
                metrics = [m for m in metrics if m in supported_metrics]

        for metric in metrics:
            logger.info(f"Visualizing {metric}")
            self.metrics_visualzer_func[metric](analyzer_items, output_dir)


if __name__ == "__main__":
    default_eval_dir = "eval_results"
    default_output_dir = "visualization_results"

    analyzer = EvalResultAnalyzer(default_eval_dir)
    analyze_results = analyzer.analyze()

    visualizer = EvalResultVisualizer()
    visualizer.visualize(analyze_results, default_output_dir)
