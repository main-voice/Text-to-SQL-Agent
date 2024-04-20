import unittest

from text_to_sql.eval.evaluator import Loader


class TestEval(unittest.TestCase):
    def test_load_dataset(self):
        loader = Loader("test_dataset.csv")
        dataset = loader.load_dataset(max_rows=10)
        self.assertEqual(len(dataset), 10)

    def test_load_eval_items(self):
        loader = Loader("test_dataset.csv")
        eval_items = loader.load_eval_items(max_rows=10)
        self.assertEqual(len(eval_items), 10)
        assert eval_items[0].question is not None
        print(eval_items[0].json())
