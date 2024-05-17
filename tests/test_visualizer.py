import unittest
from pathlib import Path

from text_to_sql.eval.models import AnalyzerItem
from text_to_sql.eval.visualizer import EvalResultAnalyzer, EvalResultVisualizer


class TestEvalVisualizer(unittest.TestCase):
    def setUp(self) -> None:
        self.test_eval_dir = Path(__file__).parent / "test_eval_results"
        self.save_dir = Path(__file__).parent / "test_visualizer_output"

    def test_analyze(self):
        analyzer = EvalResultAnalyzer(eval_dir=self.test_eval_dir)
        item = analyzer.analyze()
        assert item is not None and isinstance(item, list) and len(item) > 0 and isinstance(item[0], AnalyzerItem)

    def test_visualize(self):
        analyzer = EvalResultAnalyzer(eval_dir=self.test_eval_dir)
        items = analyzer.analyze()

        visualizer = EvalResultVisualizer()
        visualizer.visualize(items, output_dir=self.save_dir)
        visualizer.visualize_accuracy_by_hardness(items, output_dir=self.save_dir)
