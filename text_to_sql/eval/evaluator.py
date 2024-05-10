"""The Evaluator class is used to evaluate the generated SQL queries against the ground truth queries."""

import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
import psycopg2
import sentry_sdk
from sentry_sdk import capture_exception
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from tqdm import tqdm

from text_to_sql.config.settings import settings
from text_to_sql.database.db_config import DBConfig, MySQLConfig, PostgreSQLConfig
from text_to_sql.database.db_engine import DBEngine, MySQLEngine, PostgreSQLEngine
from text_to_sql.eval.models import EvalItem, EvalResultItem, SQLHardness
from text_to_sql.llm.llm_config import AzureLLMConfig, BaseLLMConfig, DeepSeekLLMConfig, LLama3LLMConfig
from text_to_sql.sql_generator.sql_generate_agent import (
    BaseSQLGeneratorAgent,
    LangchainSQLGeneratorAgent,
    SimpleSQLGeneratorAgent,
    SQLGeneratorAgent,
)
from text_to_sql.sql_generator.sql_generate_response import SQLGeneratorResponse
from text_to_sql.utils import find_bracket_index
from text_to_sql.utils.logger import get_logger

logger = get_logger(__name__)


class Loader:
    """The class for loading the evaluation items from the dataset file."""

    def __init__(self, dataset_path: str | Path):
        """Init method for the Loader class.

        Args:
            dataset_path (str): path to the dataset file.

        Raises:
            FileNotFoundError: If dataset path is not provided.
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError("Please provide a valid dataset path.")
        self.dataset_path = dataset_path

    def total_eval_items(self) -> int:
        """
        Return the total number of evaluation items in the dataset.
        """
        df = pd.read_csv(self.dataset_path)
        return len(df)

    def load_eval_items(self, start_index: int = 0, max_rows: Optional[int] = None) -> List[EvalItem]:
        """Load the evaluation items from the dataset file.

        Args:
            start_index (int, optional): Start index of the evaluation items. Defaults to 0.
            max_rows (Optional[int], optional): Max number of rows to load. Defaults to None.

        Returns:
            List[EvalItem]: The loaded evaluation items.
        """
        eval_items = []
        df = pd.read_csv(self.dataset_path)
        if max_rows:
            df = df.iloc[start_index : start_index + max_rows]

        for _, row in df.iterrows():
            eval_item = EvalItem(
                question=row["question"],
                query=row["query"],
                db_name=row["db_name"],
                query_category=row["query_category"],
                instructions=row["instructions"],
            )
            eval_items.append(eval_item)

        return eval_items


class Evaluator:
    """The main class for evaluating the generated SQL queries against the ground truth queries."""

    def __init__(
        self,
        dataset_path: str | Path,
        db_config: Optional[DBConfig] = None,
        model_type: str = "azure",
        model: str = "gpt-35-turbo",
        parallel_threads: int = 3,
        eval_method: Literal["agent", "langchain", "simple"] = "agent",
        verbose: bool = False,
    ):
        """Init method for the Evaluator class.

        Args:
            dataset_path (str): path to the dataset file.
            num_questions (int, optional): the number of questions to evaluate. Defaults to 10.
            db_config (Optional[DBConfig], optional): The database config for connecting database. Defaults to None.
            model_type (str, optional): Type of LLM to be used to evaluate. Defaults to "azure".
            model (str, optional): Model name of llm used. Defaults to "gpt-3.5".
            parallel_threads (int, optional): How many thread can be evaluated at the same time. Defaults to 3.
            verbose (bool, optional): If logging info during evaluation. Defaults to False.

        Raises:
            FileNotFoundError: If dataset path is not provided.
            ValueError: Unsupported database type or model type.
        """

        self.loader = Loader(dataset_path=dataset_path)

        if not db_config:
            raise ValueError("Database configuration is missing, please provide one at least.")

        # Init the database config
        self.db_config = db_config
        self.db_type = db_config.db_type

        # lazy initialization of the db engine (the database name could change for each question)
        self.db_engine: DBEngine | None = None

        self.model_type = model_type
        self.model = model

        self.llm_config: BaseLLMConfig = self.create_llm_config()

        self.output_folder = "eval_results"
        self.parallel_threads = parallel_threads
        self.verbose = verbose
        self.eval_method = eval_method

        self.sql_generator: BaseSQLGeneratorAgent | None = None
        self.eval_results: List[EvalResultItem] = []

    def create_llm_config(self) -> BaseLLMConfig:
        """
        create the LLM config based on the model type and model name.
        """
        supported_model_types = ["azure", "llama3", "deepseek"]
        if self.model_type not in supported_model_types:
            raise ValueError(
                f"Unsupported model type {self.model_type}. Supported model types are {supported_model_types}"
            )

        if self.model_type == "azure":
            logger.info("Creating Azure LLM Config for Evaluation...")
            if "gpt-4" in self.model:
                self.llm_config = AzureLLMConfig(model=self.model, deployment_name="gpt-4-turbo")
            else:
                self.llm_config = AzureLLMConfig(model=self.model, deployment_name="gpt-35-turbo")
        elif self.model_type == "llama3":
            logger.info("Creating Llama3 LLM Config for Evaluation...")
            self.llm_config = LLama3LLMConfig(model=self.model)
        elif self.model_type == "deepseek":
            logger.info("Creating DeepSeek LLM Config for Evaluation...")
            self.llm_config = DeepSeekLLMConfig(model=self.model)
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        return self.llm_config

    def create_sql_generator(self, db_name: Optional[str] = None):
        """
        Create the SQL generator agent based on the model type and model name.
        """
        # the SQL generator is already created for the same database
        if db_name and self.sql_generator and self.sql_generator.db_config.db_name == db_name:
            logger.info(f"The SQL Generator already connected to the database {db_name}, use the existing one")
            return self.sql_generator

        if db_name and self.db_config.db_name != db_name:
            logger.info(f"Changing the database name to {db_name} for the SQL Generator.")
            self.db_config.db_name = db_name

        if self.eval_method == "agent":
            # create the SQL generator agent, NOTICE: we don't consider time for evaluation
            self.sql_generator = SQLGeneratorAgent(
                llm_config=self.llm_config, db_config=self.db_config, verbose=self.verbose, add_current_time=False
            )
        elif self.eval_method == "langchain":
            self.sql_generator = LangchainSQLGeneratorAgent(llm_config=self.llm_config, db_config=self.db_config)
        elif self.eval_method == "simple":
            self.sql_generator = SimpleSQLGeneratorAgent(llm_config=self.llm_config, db_config=self.db_config)
        else:
            raise ValueError(f"Unsupported evaluation method {self.eval_method}")
        return self.sql_generator

    @classmethod
    def get_all_acceptable_sub_queries(cls, query: str) -> List[str]:
        """
        Extract all acceptable sub-queries from the given query (including itself).
        - split by semicolon. this is to accommodate queries where joins to other tables are also acceptable.
        - expand all column permutations if there are braces { } in it. eg:
        ```sql
            SELECT {user.id, user.name} FROM user;
        ```
        Would be expanded to:
        ```sql
            SELECT user.id FROM user;
            SELECT user.name FROM user;
            SELECT user.id, user.name FROM user;
        ```
        """
        queries = query.split(";")
        acceptable_sub_queries = []

        for q in queries:
            q = q.strip()
            if not q:
                continue

            # find acceptable sub-queries
            start_index, end_index = find_bracket_index(q)
            if (start_index, end_index) == (-1, -1):
                # only one query is acceptable
                acceptable_sub_queries.append(q)
                continue

            # expand all column permutations
            acceptable_options = q[start_index + 1 : end_index].split(",")
            acceptable_options = [option.strip() for option in acceptable_options]

            option_combinations = list(
                itertools.chain.from_iterable(
                    itertools.combinations(acceptable_options, r) for r in range(1, len(acceptable_options) + 1)
                )
            )

            for option_tuple in option_combinations:
                left_part = q[:start_index]
                right_part = q[end_index + 1 :]
                option_str = ", ".join(option_tuple)
                # change group by size depending on the number of columns
                if "GROUP BY" in right_part:
                    right_part = right_part.replace("GROUP BY {}", f"GROUP BY {option_str}")

                new_query = left_part + option_str + right_part

                acceptable_sub_queries.append(new_query)

        if len(acceptable_sub_queries) > 1:
            logger.info(f"Extracted all acceptable sub-queries: {acceptable_sub_queries}")

        return acceptable_sub_queries

    @staticmethod
    def eval_sql_hardness(sql: str) -> SQLHardness:
        """Evaluate the hardness of the SQL query.

        Args:
            sql (str): the SQL query to be evaluated.

        Returns:
            SQLHardness: The hardness level of the SQL query. An enum value.
        """
        # count the number of SQL components and nested subqueries, and aggregate functions
        number_sql_components = Evaluator.count_sql_components(sql)
        number_nested_subqueries = Evaluator.count_nested_subqueries(sql)
        number_aggregate = Evaluator.count_aggregate(sql)

        # evaluate the hardness of the SQL query
        if number_sql_components <= 1 and number_nested_subqueries == 0 and number_aggregate == 0:
            return SQLHardness.EASY
        if (number_aggregate <= 2 and number_sql_components <= 1 and number_nested_subqueries == 0) or (
            number_sql_components <= 2 and number_aggregate < 2 and number_nested_subqueries == 0
        ):
            return SQLHardness.MEDIUM

        if (
            (number_aggregate > 2 and number_sql_components <= 2 and number_nested_subqueries == 0)
            or (2 < number_sql_components <= 3 and number_aggregate <= 2 and number_nested_subqueries == 0)
            or (number_sql_components <= 1 and number_aggregate == 0 and number_nested_subqueries <= 1)
        ):
            return SQLHardness.HARD

        return SQLHardness.ULTRA

    @staticmethod
    def count_sql_components(sql: str) -> int:
        """Count the number of SQL components in the SQL query, including where, group by, order by, etc.

        Args:
            sql (str): the SQL query to be counted.

        Returns:
            int: All the SQL components in the SQL query.
        """
        # count keywords
        count_keywords = len(re.findall(r"\b(WHERE|GROUP BY|ORDER BY|LIMIT|HAVING|JOIN|AND)\b", sql, re.IGNORECASE))
        # count 'or'
        count_or = len(re.findall(r"\bOR\b", sql, re.IGNORECASE))
        # count 'like'
        count_like = len(re.findall(r"\bLIKE\b", sql, re.IGNORECASE))

        return count_keywords + count_or + count_like

    @staticmethod
    def count_nested_subqueries(sql: str) -> int:
        """
        Count the number of nested subqueries in the SQL query.
        """
        count_nested_subqueries = len(
            re.findall(r"\((SELECT|SELECT .+ UNION|SELECT .+ INTERSECT|SELECT .+ EXCEPT)", sql, re.IGNORECASE)
        )
        return count_nested_subqueries

    @staticmethod
    def count_aggregate(sql: str) -> int:
        """
        Count the number of aggregate functions in the SQL query.
        """
        count_agg = len(re.findall(r"\b(COUNT|SUM|AVG|MIN|MAX)\b", sql, re.IGNORECASE))
        return count_agg

    def load_previous_eval_results(self, previous_eval_file: str) -> List[EvalResultItem]:
        """Load previous evaluation results from file.

        Args:
            previous_eval_file (str): Path to the previous evaluation result file.
        """
        previous_eval_file = Path(__file__).parent / Path(self.output_folder) / Path(previous_eval_file)
        if not previous_eval_file.exists():
            logger.warning(f"Previous evaluation file {previous_eval_file} does not exist.")
            return []

        # load the previous evaluation results from the JSON file
        with open(previous_eval_file, "r", encoding="utf-8") as f:
            eval_results = json.load(f)

        pre_eval_results = [EvalResultItem.parse_obj(item) for item in eval_results]
        logger.info(f"Loaded {len(pre_eval_results)} previous evaluation results from {previous_eval_file}.")
        return pre_eval_results

    def eval(
        self, num_questions: int = 10, start_index: int = 0, previous_eval_file: Optional[str] = None
    ) -> List[EvalResultItem]:
        """Evaluate the generated SQL queries against the ground truth queries.

        Args:
            num_questions (int, optional): Number of questions to evaluate. Defaults to 10.
            start_index (int, optional): Start index of the questions to evaluate. Defaults to 0.
            previous_eval_file (Optional[str], optional): Path to the previous evaluation result file.
                                                        If provided, will do incremental evaluation. Defaults to None.

        Returns:
            List[EvalResultItem]: The evaluated result items.
        """

        if previous_eval_file:
            pre_eval_results = self.load_previous_eval_results(previous_eval_file)

        max_num_questions = self.loader.total_eval_items()
        if start_index < 0 or start_index >= max_num_questions:
            raise ValueError(f"Start index {start_index} is out of the range of the dataset.")

        if num_questions + start_index > max_num_questions:
            logger.warning(
                f"Number of questions {num_questions} is bigger than the maximum questions allowed {max_num_questions},\
                  will be using the maximum number of questions allowed."
            )
            num_questions = max_num_questions - start_index

        # first we have a dataframe with the questions and the ground truth queries
        eval_items: List[EvalItem] = self.loader.load_eval_items(max_rows=num_questions, start_index=start_index)
        logger.info(f"Loaded {len(eval_items)} evaluation items starting from index {start_index}.")
        self.eval_results = []

        for eval_item in tqdm(eval_items, desc="Evaluating...", total=len(eval_items)):
            if previous_eval_file:
                # Check if the current eval_item exists in previous results
                previous_result = next(
                    (
                        item
                        for item in pre_eval_results
                        if item.question == eval_item.question and item.db_name == eval_item.db_name
                    ),
                    None,
                )
                if previous_result and previous_result.is_correct:
                    # If the previous result is correct, skip evaluation
                    logger.info(f"Skip evaluation: Question: {eval_item.question} is already correct.")
                    self.eval_results.append(previous_result)
                    continue

            logger.info(f"Start evaluating question: {eval_item.question}, the database is {eval_item.db_name}...")

            item_eval_result = self.eval_single_item(eval_item)
            self.eval_results.append(item_eval_result)

        return self.eval_results

    def eval_single_item(self, eval_item: EvalItem | EvalResultItem) -> EvalResultItem:
        """Evaluate a single evaluation item.

        Args:
            eval_item (EvalItem | EvalResultItem): The evaluation item to evaluate.

        Returns:
            EvalResultItem: The evaluation result for the given eval_item.
        """
        self.create_sql_generator(db_name=eval_item.db_name)

        # record the time for evaluation
        instructions = None
        if eval_item.instructions and eval_item.instructions != "nan":
            instructions = eval_item.instructions

        start_time = time.time()
        sql_generator_response: SQLGeneratorResponse = self.sql_generator.generate_sql(
            user_query=eval_item.question,
            instructions=instructions,
            single_line_format=True,
            verbose=self.verbose,
        )
        end_time = time.time()
        eval_duration_time = round(end_time - start_time, 2)

        sql_hardness = self.eval_sql_hardness(eval_item.golden_query)
        result_item = EvalResultItem(
            question=eval_item.question,
            golden_query=eval_item.golden_query,
            db_name=eval_item.db_name,
            query_category=eval_item.query_category,
            instructions=eval_item.instructions,
            generated_query=sql_generator_response.generated_sql,
            token_usage=sql_generator_response.token_usage,
            hardness=sql_hardness,
            eval_duration=eval_duration_time,
        )

        if not sql_generator_response.generated_sql or sql_generator_response.error:
            if sql_generator_response.error:
                result_item.error_detail = sql_generator_response.error
            else:
                result_item.error_detail = "Failed to generate the query."
            logger.error(
                f"Failed to generate query for question: {eval_item.question},\
                the error is {sql_generator_response.error}."
            )
            return result_item

        logger.info(f"Generated query: {sql_generator_response.generated_sql}.")

        logger.info("Evaluating the correctness of the generated query...")
        self.check_query_correctness(result_item)

        if result_item.is_correct:
            logger.info("The generated query is correct.")
        else:
            logger.error("The generated query is incorrect.")

        return result_item

    def check_query_correctness(self, result_item: EvalResultItem) -> EvalResultItem:
        """Check the correctness of the generated query with the golden queries. Will check the execution correctness.
        # TODO: Add more evaluation metrics.

        Args:
            result_item (EvalResultItem): the result item to be checked.

        Returns:
            EvalResultItem: the evaluated result item, with the comparison result.
        """
        # check if the generated query produces the same result as the golden query exactly
        result_item.exec_correct = self.check_exec_correctness(result_item)
        result_item.is_correct = result_item.exec_correct
        return result_item

    def check_exec_correctness(self, result_item: EvalResultItem):
        """Check the execution correctness of the generated queries.

        Args:
            result_item (EvalResultItem): the result item to be checked.
        """
        generated_query = result_item.generated_query
        golden_query = result_item.golden_query
        db_name = result_item.db_name
        question = result_item.question

        acceptable_sub_queries: List[str] = self.get_all_acceptable_sub_queries(golden_query)
        if generated_query in acceptable_sub_queries:
            # exact match
            logger.info("The generated query is an exact match.")
            return True

        if not self.db_engine or self.db_engine.db_config.db_name != db_name:
            # if the db engine is not initialized or the db name is different, we need to reconnect
            self.get_db_engine(db_name=db_name)

        # execute the generated query
        try:
            my_result = self.query_db(generated_query, db_name)
            logger.debug(f"My SQL result: {my_result}")

            for acceptable_query in acceptable_sub_queries:
                acceptable_result = self.query_db(acceptable_query, db_name)

                if self.compare_df(
                    acceptable_result, my_result, result_item.query_category, question, golden_query, generated_query
                ):
                    logger.info(f"The generated query is correct. One of acceptable result: \n{acceptable_result}\n")
                    return True

            warning_message = "Can not match any golden query results"
            logger.warning(warning_message)
            result_item.error_detail = warning_message
            return False

        except Exception as e:
            result_item.error_detail = str(e)
            return False

    def compare_df(
        self,
        df_golden: pd.DataFrame,
        df_generated: pd.DataFrame,
        query_category: str,
        question: str,
        query_golden: str = None,
        query_generated: str = None,
    ) -> bool:
        """Compares two dataframes

        query_gold and query_gen are the original queries that generated the respective dataframes.

        Args:
            df_golden (pd.DataFrame): The golden dataframe
            df_generated (pd.DataFrame): The generated dataframe from the generated query
            query_category (str): The category of the query
            question (str): The original question
            query_golden (str): The original query that generated the golden dataframe
            query_generated (str): The original query that generated the generated dataframe

        Returns:
            bool: True if the dataframes are equal, False otherwise
        """
        if df_golden.equals(df_generated):
            return True

        # ignore the column name
        if df_golden.shape == df_generated.shape and (df_golden.values == df_generated.values).all():
            return True

        # normalize the dataframes
        df_golden = self.normalize_pd_table(df_golden, query_category, question, query_golden)
        df_generated = self.normalize_pd_table(df_generated, query_category, question, query_generated)

        if df_golden.shape == df_generated.shape and (df_golden.values == df_generated.values).all():
            return True

        return False

    def normalize_pd_table(self, df: pd.DataFrame, query_category: str, question: str, sql: str = None) -> pd.DataFrame:
        """
        Normalizes a dataframe by:
        1. removing all duplicate rows
        2. sorting columns in alphabetical order
        3. sorting rows using values from first column to last (if query_category is not 'order_by' and question
        does not ask for ordering)
        4. resetting index
        """
        # remove duplicate rows, if any
        df = df.drop_duplicates()

        # sort columns in alphabetical order of column names
        sorted_df = df.reindex(sorted(df.columns), axis=1)

        # check if query_category is 'order_by' and if question asks for ordering
        has_order_by = False
        pattern = re.compile(r"\b(order|sort|arrange)\b", re.IGNORECASE)
        in_question = re.search(pattern, question.lower())  # true if contains
        if query_category == "order_by" or in_question:
            has_order_by = True

            if sql:
                # determine which columns are in the ORDER BY clause of the sql generated, using regex
                pattern = re.compile(r"ORDER BY[\s\S]*", re.IGNORECASE)
                order_by_clause = re.search(pattern, sql)
                if order_by_clause:
                    order_by_clause = order_by_clause.group(0)
                    # get all columns in the ORDER BY clause, by looking at the text between ORDER BY and
                    # the next semicolon, comma, or parantheses
                    pattern = re.compile(r"(?<=ORDER BY)(.*?)(?=;|,|\)|$)", re.IGNORECASE)
                    order_by_columns = re.findall(pattern, order_by_clause)
                    order_by_columns = order_by_columns[0].split() if order_by_columns else []
                    order_by_columns = [col.strip().rsplit(".", 1)[-1] for col in order_by_columns]

                    ascending = False
                    # if there is a DESC or ASC in the ORDER BY clause, set the ascending to that
                    if "DESC" in [i.upper() for i in order_by_columns]:
                        ascending = False
                    elif "ASC" in [i.upper() for i in order_by_columns]:
                        ascending = True

                    # remove whitespace, commas, and parantheses
                    order_by_columns = [col.strip() for col in order_by_columns]
                    order_by_columns = [col.replace(",", "").replace("(", "") for col in order_by_columns]
                    order_by_columns = [
                        i
                        for i in order_by_columns
                        if i.lower() not in ["desc", "asc", "nulls", "last", "first", "limit"]
                    ]

                    # get all columns in sorted_df that are not in order_by_columns
                    other_columns = [i for i in sorted_df.columns.tolist() if i not in order_by_columns]

                    # only choose order_by_columns that are in sorted_df
                    order_by_columns = [i for i in order_by_columns if i in sorted_df.columns.tolist()]
                    sorted_df = sorted_df.sort_values(by=order_by_columns + other_columns, ascending=ascending)

        if not has_order_by:
            # sort rows using values from first column to last
            sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))

        # reset index
        sorted_df = sorted_df.reset_index(drop=True)
        return sorted_df

    def save_output(
        self, eval_type: Literal["smoke", "prod"] = "smoke", eval_result: List[EvalResultItem] = None
    ) -> str:
        """Save the evaluation output to a JSON file.

        Args:
            eval_type (Literal["smoke", "prod"], optional): The evaluation type. Defaults to "smoke".
            eval_result (List[EvalResultItem]): the evaluation result to be saved, if not provided

        Returns:
            str: path to the saved file, will connect the prefix with the current time
        """
        if not eval_result:
            return ""
        save_folder = Path(__file__).parent / self.output_folder / self.model_type
        save_folder.mkdir(parents=True, exist_ok=True)

        # TODO: add time to the file name when the evaluation is stable
        # save_path = f"{eval_type}_{self.eval_method}_{self.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_path = f"{eval_type}_{self.eval_method}_{self.model}.json"
        save_path = save_folder / save_path

        # save the evaluation result to a json file
        eval_result_json = [item.dict() for item in eval_result]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(eval_result_json, f, indent=4)
        logger.info(f"Saved the evaluation result to {save_path}.")
        return save_path.as_posix()

    def get_db_engine(self, db_name: Optional[str] = None) -> DBEngine:
        """Connect to the database and return the db engine.

        Args:
            db_name (Optional[str], optional): The database name to be connected. Defaults to None.

        Raises:
            ValueError: If no database configuration is provided before calling this method,
            Or unsupported database type.

        Returns:
            DBEngine: the database engine
        """
        # if db_name is None, we return the db engine if it is already initialized
        if db_name is None and self.db_engine:
            logger.warning("No database name provided, using the existing db engine.")
            return self.db_engine

        # if db_name is provided and the db engine is already initialized with the same db_name, we return the db engine
        if db_name and self.db_engine and self.db_engine.db_config.db_name == db_name:
            logger.info(f"The database {db_name} engine is already initialized, use the existing one.")
            return self.db_engine

        # first call, so we need to initialize the db engine
        if not self.db_config and not self.db_type:
            raise ValueError("Database configuration is missing.")

        # Create a new db engine if the db_name is different from the existing one
        # now the db_type is either the one provided explicitly or the one in the config

        if isinstance(self.db_config, PostgreSQLConfig):
            logger.info("Connecting to PostgreSQL database...")

            self.db_config = self.db_config if self.db_config.db_name == db_name else PostgreSQLConfig(db_name=db_name)
            self.db_engine = PostgreSQLEngine(self.db_config)

        elif isinstance(self.db_config, MySQLConfig):
            logger.info("Connecting to MySQL database...")
            self.db_config = self.db_config if self.db_config.db_name == db_name else MySQLConfig(db_name=db_name)
            self.db_engine = MySQLEngine(self.db_config)
        else:
            raise ValueError(f"Unsupported database type {self.db_type}")

        return self.db_engine

    def query_db(self, query: str, db_name: str) -> pd.DataFrame:
        """Query the database.

        Args:
            query (str): the query to be executed
            db_name (str): the database name to be connected
            timeout (int, optional): timeout for the query execution. Defaults to 5.

        Returns:
            dict: The query result
        """
        engine = self.get_db_engine(db_name)
        if not engine:
            raise ValueError("Failed to connect to the database.")
        if not engine.connection:
            engine.connect_db(db_name=db_name)

        try:
            query_result = engine.execute(statement=query, is_pd=True)
            # the query result is a pandas dataframe
            return query_result
        except DBAPIError as e:
            logger.error(f"DBAPIError when executing the query: error is {e}, query is {query}")
            orig = e.orig
            if isinstance(orig, psycopg2.Error):
                logger.error(f"It's origin is psycopg2 Error in executing the query: {orig}")
            capture_exception(e)
            raise e
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemyError when executing the query: error is {e}, query is {query}")
            capture_exception(e)
            raise e
        except psycopg2.Error as e:
            logger.error(f"psycopg2.Error in executing the query: error is {e}, query is {query}")
            capture_exception(e)
            raise e
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Unknown error in executing the query: error is {e}, query is {query}")
            capture_exception(e)
            raise e


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Evaluate the generated SQL queries.")

    # data related arguments
    arg_parser.add_argument("--dataset_path", type=str, help="Path to the dataset file.")
    arg_parser.add_argument("--num_questions", type=int, default=10, help="Number of questions to evaluate.")
    arg_parser.add_argument("--start_index", type=int, default=0, help="Start index of the questions to evaluate.")
    arg_parser.add_argument("--db_type", type=str, default="postgres", help="Type of the database.")

    # llm related arguments
    arg_parser.add_argument("--model_type", type=str, help="Type of the LLM model.", choices=["azure", "llama3"])
    arg_parser.add_argument("--model", type=str, help="Model name of the LLM.")

    # evaluation related arguments
    arg_parser.add_argument("--output_prefix", type=str, default="eval_output", help="Prefix path to the output file.")
    # arg_parser.add_argument("--parallel_threads", type=int, default=3, help="Number of parallel threads.")
    arg_parser.add_argument(
        "--eval_method", type=str, help="Evaluation method.", choices=["agent", "langchain", "simple"]
    )
    arg_parser.add_argument(
        "--pre_eval_result_file",
        type=str,
        default="",
        help="[Optional] Path to an existing evaluation result file, will update the existing file.",
    )

    arg_parser.add_argument(
        "--eval_type",
        type=str,
        choices=["smoke", "prod"],
        help="Evaluation type. smoke for quick test evaluation, prod for full evaluation.",
    )

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN.get_secret_value(),
        environment=settings.ENVIRONMENT,
    )

    args = arg_parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = Path(__file__).parent / "questions_gen_postgres.csv"

    # azure evaluation
    default_model_config = {
        "model_type": "azure",
        "model": "gpt-35-turbo",
    }
    if not args.model_type and not args.model:
        args.model_type = default_model_config["model_type"]
        args.model = default_model_config["model"]

    if not args.eval_method:
        raise ValueError("Please provide the evaluation method.")

    evaluator = Evaluator(
        db_config=PostgreSQLConfig(),
        model_type=args.model_type,
        model=args.model,
        dataset_path=args.dataset_path,
        verbose=True,
        eval_method=args.eval_method,
    )
    try:
        results = evaluator.eval(
            num_questions=args.num_questions, start_index=args.start_index, previous_eval_file=args.pre_eval_result_file
        )
        evaluator.save_output(args.eval_type, results)
    except Exception as error:  # pylint: disable=broad-except
        logger.error(f"Error in evaluating the generated SQL queries: {error}")
        capture_exception(error)
        evaluator.save_output(args.eval_type)
