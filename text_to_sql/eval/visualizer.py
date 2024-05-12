"""This module contains classes to analyze and visualize the evaluation results."""

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from text_to_sql.eval.models import AccuracyItem, AnalyzerItem, BoxPlotItem, EvalResultItem, SQLHardness
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class EvalResultAnalyzer:
    """Analyze evaluation results and visualize the results."""

    def __init__(self, eval_dir: str):
        self.eval_dir = Path(__file__).parent / eval_dir

    def analyze_accuracy(self, eval_results: List[EvalResultItem]) -> AccuracyItem:
        """Analyze the accuracy of the evaluation results, including overall and by different hardness level."""
        accuracy_all = sum(1 for r in eval_results if r.is_correct) / len(eval_results)
        easy_item_count = len([r for r in eval_results if r.hardness == SQLHardness.EASY])
        easy_accuracy = 0
        if easy_item_count > 0:
            logger.debug(f"EASY Hardness item number count: {easy_item_count}.")
            easy_accuracy = (
                sum(1 for r in eval_results if r.hardness == SQLHardness.EASY and r.is_correct) / easy_item_count
            )
        else:
            logger.debug("No easy item found.")

        medium_item_count = len([r for r in eval_results if r.hardness == SQLHardness.MEDIUM])
        medium_accuracy = 0
        if medium_item_count > 0:
            logger.debug(f"MEDIUM Hardness item number count: {medium_item_count}.")
            medium_accuracy = (
                sum(1 for r in eval_results if r.hardness == SQLHardness.MEDIUM and r.is_correct) / medium_item_count
            )
        else:
            logger.debug("No medium item found.")

        hard_item_count = len([r for r in eval_results if r.hardness == SQLHardness.HARD])
        hard_accuracy = 0
        if hard_item_count > 0:
            logger.debug(f"HARD Hardness item number count: {hard_item_count}.")
            hard_accuracy = (
                sum(1 for r in eval_results if r.hardness == SQLHardness.HARD and r.is_correct) / hard_item_count
            )
        else:
            logger.debug("No hard item found.")

        ultra_item_count = len([r for r in eval_results if r.hardness == SQLHardness.ULTRA])
        ultra_accuracy = 0
        if ultra_item_count > 0:
            logger.debug(f"ULTRA Hardness item number count: {ultra_item_count}.")
            ultra_accuracy = (
                sum(1 for r in eval_results if r.hardness == SQLHardness.ULTRA and r.is_correct) / ultra_item_count
            )
        else:
            logger.debug("No ultra item found.")

        return AccuracyItem(
            accuracy_all=accuracy_all,
            easy_item_count=easy_item_count,
            easy_accuracy=easy_accuracy,
            medium_item_count=medium_item_count,
            medium_accuracy=medium_accuracy,
            hard_item_count=hard_item_count,
            hard_accuracy=hard_accuracy,
            ultra_item_count=ultra_item_count,
            ultra_accuracy=ultra_accuracy,
        )

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

                accuracy_item = self.analyze_accuracy(eval_results)

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
                        accuracy=accuracy_item,
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
        self.colors = [
            "#8ECFC9",
            "#FFBE7A",
            "#FA7F6F",
            "#82B0D2",
            "#BEB8DC",
            "#E7DAD2",
        ]
        self.transparency = 0.5
        self.dpi = 350
        self.figsize = (10, 6)

    def visualize_accuracy(self, analyzer_items: List[AnalyzerItem], output_dir: str):
        """
        Visualize the accuracy of the evaluation results.
        """
        output_dir = self.preprocess_output_dir(output_dir)

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        bar_width = 0.15
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
                accuracy = item.accuracy.accuracy_all if item and item.accuracy.accuracy_all else 0
                y.append(accuracy)

            plt.bar(x + i * bar_width, y, width=bar_width, label=model, color=self.colors[i % len(self.colors)])

        plt.xticks(x + bar_width, methods)
        plt.xlabel("Eval Methods")
        plt.ylabel("Accuracy")
        plt.title("Comparison of Accuracy")

        plt.legend()
        plt.tight_layout()

        output_file = output_dir.joinpath("accuracy.png")
        plt.savefig(output_file)
        plt.close()

    def visualize_accuracy_by_hardness(self, analyzer_items: List[AnalyzerItem], output_dir: str | Path):
        """
        Visualize accuracy by difficulty level for each model and method.
        """
        output_dir = self.preprocess_output_dir(output_dir)

        hardness_levels = ["easy", "medium", "hard", "ultra"]
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        bar_width = 0.8 / len(hardness_levels)  # Adjust bar width based on the number of hardness levels

        # Prepare x-axis labels and positions for bars
        models_methods = sorted(set((item.llm_model, item.method) for item in analyzer_items))
        x = np.arange(len(models_methods))

        # Collect data for each hardness level
        for index, hardness in enumerate(hardness_levels):
            accuracies = []
            for model, method in models_methods:
                item = next(
                    (item for item in analyzer_items if item.llm_model == model and item.method == method), None
                )
                if item:
                    accuracy = getattr(item.accuracy, f"{hardness}_accuracy", 0)
                else:
                    accuracy = 0
                accuracies.append(accuracy)

            # Plot bars for the current hardness level
            plt.bar(
                x + index * bar_width,
                accuracies,
                width=bar_width,
                label=hardness.capitalize(),
                color=self.colors[index],
            )

        # Customize chart and save
        plt.xticks(
            x + bar_width * len(hardness_levels) / 2,
            [f"{model} & {method}" for model, method in models_methods],
            rotation=45,
        )
        plt.xlabel("Model + Eval Method")
        plt.ylabel("Accuracy")
        plt.title("Comparison of Accuracy by Hardness Levels")
        plt.legend(title="Hardness")

        plt.tight_layout()
        output_file = output_dir.joinpath("accuracy_by_hardness.png")
        plt.savefig(output_file)
        plt.close()

    def visualize_token_usage(self, analyzer_items: List[AnalyzerItem], output_dir: str | Path):
        """
        Visualize the token usage of the evaluation results.
        """
        output_dir = self.preprocess_output_dir(output_dir)

        plt.figure(figsize=self.figsize, dpi=self.dpi)
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

        # set colors for different models (same models have the same color)
        model_colors = {model: self.colors[i % len(self.colors)] for i, model in enumerate(models)}

        box_plots = plt.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=False,
            showmeans=True,
            meanline=True,
            flierprops={"marker": "o", "markersize": 5},
            medianprops={"linewidth": 1.5},
            meanprops={"linewidth": 1.5},
        )

        for plot, color in zip(box_plots["boxes"], [model_colors[model] for model in models for _ in methods]):
            plot.set_facecolor(color)
            plot.set_alpha(self.transparency)

            plot.set_edgecolor(color)

        # set colors for medians and means same as the box color
        for median, color in zip(box_plots["medians"], [model_colors[model] for model in models for _ in methods]):
            median.set_color(color)

        for mean, color in zip(box_plots["means"], [model_colors[model] for model in models for _ in methods]):
            mean.set_color(color)

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
        output_dir = self.preprocess_output_dir(output_dir)

        plt.figure(figsize=self.figsize, dpi=self.dpi)
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

        # set colors for different models (same models have the same color)
        model_colors = {model: self.colors[i % len(self.colors)] for i, model in enumerate(models)}

        box_plots = plt.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=False,
            showmeans=True,
            meanline=True,
            flierprops={"marker": "o", "markersize": 5},
            medianprops={"linewidth": 1.5},
            meanprops={"linewidth": 1.5},
        )
        for plot, color in zip(box_plots["boxes"], [model_colors[model] for model in models for _ in methods]):
            plot.set_facecolor(color)
            plot.set_alpha(self.transparency)

            plot.set_edgecolor(color)

        # set colors for medians and means same as the box color
        for median, color in zip(box_plots["medians"], [model_colors[model] for model in models for _ in methods]):
            median.set_color(color)

        for mean, color in zip(box_plots["means"], [model_colors[model] for model in models for _ in methods]):
            mean.set_color(color)

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
        output_dir = self.preprocess_output_dir(output_dir)

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

    def preprocess_output_dir(self, output_dir: str | Path):
        """Preprocess the output directory."""
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


if __name__ == "__main__":
    default_eval_dir = "eval_results"
    default_output_dir = "visualization_results"

    analyzer = EvalResultAnalyzer(default_eval_dir)
    analyze_results = analyzer.analyze()

    visualizer = EvalResultVisualizer()
    visualizer.visualize(analyze_results, default_output_dir)
    visualizer.visualize_accuracy_by_hardness(analyze_results, default_output_dir)
