import unittest

from pathlib import Path

from text_to_sql.eval.evaluator import Evaluator, Loader


class TestEval(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = Evaluator(dataset_path=Path(__file__).parent / "test_dataset.csv", db_type="postgres")

    def test_load_dataset(self):
        with self.assertRaises(FileNotFoundError):
            loader = Loader("test_dataset.csv")

        abs_path = Path(__file__).parent / "test_dataset.csv"

        loader = Loader(abs_path)
        dataset = loader.load_eval_items(max_rows=10)
        self.assertEqual(len(dataset), 10)
        assert dataset[0].question is not None
        print(dataset[0].json())

        self.assertEqual(len(dataset), 10)

    def test_exec_correctness(self):
        """
        Test if the generated SQL is equal to the expected SQL
        """
        sql1: str = "select * from publication where year >= 2010;"
        sql2: str = "select * from publication where year >= 2010;"
        self.assertTrue(self.evaluator.check_exec_correctness(sql1, sql2, db_name="academic"))

        generated_sql: str = """SELECT conference.name, count(publication.pid) AS publication_count FROM publication \
        JOIN conference ON publication.cid = conference.cid WHERE publication.year >= extract(YEAR FROM CURRENT_DATE)\
         - 15 GROUP BY conference.name ORDER BY publication_count DESC LIMIT 1;"""
        expected_sql: str = """select c.name as conference_name, count(p.pid) as publication_count from conference c \
        join publication p on c.cid = p.cid where p.year >= extract(year from current_date) - 15 group by c.name order\
         by count(p.pid) desc limit 1;"""

        self.assertTrue(self.evaluator.check_exec_correctness(generated_sql, expected_sql, db_name="academic"))
