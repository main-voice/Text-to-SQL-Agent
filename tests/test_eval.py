import unittest
from pathlib import Path

from text_to_sql.database.db_config import PostgreSQLConfig
from text_to_sql.eval.evaluator import Evaluator, Loader
from text_to_sql.eval.models import EvalResultItem


class TestEval(unittest.TestCase):
    def setUp(self) -> None:
        db_config = PostgreSQLConfig(db_name="academic")
        self.evaluator = Evaluator(dataset_path=Path(__file__).parent / "test_dataset.csv", db_config=db_config)

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
        test_same_sql: str = "select * from publication where year >= 2010;"
        eval_result_item = EvalResultItem(
            question="test question",
            db_name="academic",
            query_category="select",
            query=test_same_sql,
            generated_query=test_same_sql,
        )
        self.assertTrue(self.evaluator.check_exec_correctness(eval_result_item))

        generated_sql: str = """SELECT conference.name, count(publication.pid) AS publication_count FROM publication \
        JOIN conference ON publication.cid = conference.cid WHERE publication.year >= extract(YEAR FROM CURRENT_DATE)\
         - 15 GROUP BY conference.name ORDER BY publication_count DESC LIMIT 1;"""
        expected_sql: str = """select c.name as conference_name, count(p.pid) as publication_count from conference c \
        join publication p on c.cid = p.cid where p.year >= extract(year from current_date) - 15 group by c.name order\
         by count(p.pid) desc limit 1;"""

        eval_result_item = EvalResultItem(
            question="test question",
            db_name="academic",
            query_category="select",
            query=expected_sql,
            generated_query=generated_sql,
        )

        self.assertTrue(self.evaluator.check_exec_correctness(eval_result_item))

    def test_error_handle(self):
        """
        Test some error cases when evaluating the SQL
        """
        error_sql_divide_zero: str = (
            "select d.did,     (select count(distinct p.pid) from domain_publication dp \
            join publication p on dp.pid = p.pid where dp.did = d.did) /     (select count(distinct k.kid) from \
            domain_keyword dk join keyword k on dk.kid = k.kid where dk.did = d.did) as ratio from domain d;"
        )
        refer_sql = "SELECT domain_publication.did, CAST(COUNT(DISTINCT domain_publication.pid) AS FLOAT)\
             / NULLIF(COUNT(DISTINCT domain_keyword.kid), 0) AS publication_to_keyword_ratio FROM domain_publication \
            LEFT JOIN domain_keyword ON domain_publication.did = domain_keyword.did GROUP BY domain_publication.did \
            ORDER BY publication_to_keyword_ratio DESC NULLS LAST;"
        eval_result_item = EvalResultItem(
            question="What is the ratio of the total number of publications to the \
            total number of keywords within each domain ID? Show all domain IDs.",
            db_name="academic",
            query_category="select",
            query=refer_sql,
            generated_query=error_sql_divide_zero,
        )
        self.assertFalse(self.evaluator.check_exec_correctness(eval_result_item))

    def test_query_db(self):
        """Test if the query is correct"""
        test_sql = "SELECT DISTINCT author.name FROM author JOIN writes ON author.aid = writes.aid JOIN publication ON \
        writes.pid = publication.pid JOIN domain_publication ON publication.pid = domain_publication.pid JOIN DOMAIN ON\
         domain_publication.did = domain.did WHERE domain.name ilike '%computer%science%'"
        pd_result = self.evaluator.query_db(test_sql, db_name="academic")
        assert pd_result is not None and pd_result.values is not None

    def test_expand_sub_queries(self):
        """Test if the sub-queries are expanded correctly"""
        query1 = "SELECT * FROM persons WHERE persons.age > 25"

        assert self.evaluator.get_all_acceptable_sub_queries(query1) == [query1]
        query2 = "SELECT persons.name FROM persons WHERE persons.age > 25 GROUP BY 1"
        assert self.evaluator.get_all_acceptable_sub_queries(query2) == [query2]
        query3 = "SELECT {persons.name,persons.id} FROM persons WHERE persons.age > 25"
        option1 = "SELECT persons.name FROM persons WHERE persons.age > 25"
        option2 = "SELECT persons.id FROM persons WHERE persons.age > 25"
        option3 = "SELECT persons.name, persons.id FROM persons WHERE persons.age > 25"
        assert self.evaluator.get_all_acceptable_sub_queries(query3) == [option1, option2, option3]
        query4 = "SELECT {persons.name,persons.id} FROM persons WHERE persons.age > 25 GROUP BY {}"
        option1 = "SELECT persons.name FROM persons WHERE persons.age > 25 GROUP BY persons.name"
        option2 = "SELECT persons.id FROM persons WHERE persons.age > 25 GROUP BY persons.id"
        option3 = (
            "SELECT persons.name, persons.id FROM persons WHERE persons.age > 25 GROUP BY persons.name, persons.id"
        )
        assert self.evaluator.get_all_acceptable_sub_queries(query4) == [option1, option2, option3]
