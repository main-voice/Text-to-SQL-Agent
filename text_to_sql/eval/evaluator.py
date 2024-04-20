"""The Evaluator class is used to evaluate the generated SQL queries against the ground truth queries."""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from text_to_sql.database.db_config import DBConfig, MySQLConfig, PostgreSQLConfig
from text_to_sql.database.db_engine import DBEngine, MySQLEngine, PostgreSQLEngine
from text_to_sql.eval.models import EvalItem, EvalResultItem
from text_to_sql.llm.llm_config import AzureLLMConfig
from text_to_sql.sql_generator.sql_generate_agent import SQLGeneratorAgent
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

    def load_eval_items(self, max_rows: Optional[int] = None) -> List[EvalItem]:
        """Load the evaluation items from the dataset file.

        Args:
            max_rows (Optional[int], optional): Max number of rows to load. Defaults to None.

        Returns:
            List[EvalItem]: The loaded evaluation items.
        """
        eval_items = []
        df = pd.read_csv(self.dataset_path)
        if max_rows:
            df = df.head(max_rows)

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
        num_questions: int = 10,
        max_num_questions: Optional[int] = None,
        db_type: Optional[str] = None,
        db_config: Optional[DBConfig] = None,
        model_type: str = "azure",
        model: str = "gpt-35-turbo",
        output_prefix: str = "eval_output",
        parallel_threads: int = 3,
        verbose: bool = False,
    ):
        """Init method for the Evaluator class.

        Args:
            dataset_path (str): path to the dataset file.
            num_questions (int, optional): the number of questions to evaluate. Defaults to 10.
            max_num_questions (Optional[int], optional): Max number questions to evaluate. Defaults to None.
            db_type (Optional[str], optional): type of database type. Defaults to None.
            db_config (Optional[DBConfig], optional): The database config for connecting database. Defaults to None.
            model_type (str, optional): Type of LLM to be used to evaluate. Defaults to "azure".
            model (str, optional): Model name of llm used. Defaults to "gpt-3.5".
            output_prefix (str, optional): Prefix path to the output evaluation result. Defaults to "eval_output".
            parallel_threads (int, optional): How many thread can be evaluated at the same time. Defaults to 3.
            verbose (bool, optional): If logging info during evaluation. Defaults to False.

        Raises:
            FileNotFoundError: If dataset path is not provided.
            ValueError: Unsupported database type or model type.
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError("Please provide a valid dataset path.")
        self.dataset_path = dataset_path
        self.loader = Loader(dataset_path=dataset_path)

        if max_num_questions and num_questions > max_num_questions:
            logger.warning(
                f"Number of questions provided {num_questions} is greater than the maximum number \
            of questions allowed {max_num_questions}, will be using the maximum number of questions allowed."
            )
            num_questions = max_num_questions
        self.num_questions = num_questions

        if not db_type and not db_config:
            raise ValueError("Database type or configuration is missing, please provide one at least.")

        supported_db_types = ["postgres", "mysql"]
        if db_type and db_type not in supported_db_types:
            raise ValueError(f"Unsupported database type {db_type}. Supported database types are {supported_db_types}")
        if db_type and db_config and db_type != db_config.db_type:
            raise ValueError(
                f"Database type {db_type} does not match the database type in the config {db_config.db_type}"
            )

        # Init the database config
        if db_config:
            self.db_type = db_config.db_type
            self.db_config = db_config
        else:
            self.db_type = db_type
            if self.db_type == "postgres":
                self.db_config = PostgreSQLConfig()
            elif self.db_type == "mysql":
                self.db_config = MySQLConfig()

        # lazy initialization of the db engine (the database name could change for each question)
        self.db_engine: DBEngine | None = None

        supported_model_types = ["azure"]
        if model_type not in supported_model_types:
            raise ValueError(f"Unsupported model type {model_type}. Supported model types are {supported_model_types}")
        self.model_type = model_type
        self.model = model
        if self.model_type == "azure":
            logger.info("Creating Azure LLM Config for Evaluation...")
            self.llm_config = AzureLLMConfig(model=self.model)

        self.output_prefix = output_prefix
        self.parallel_threads = parallel_threads
        self.verbose = verbose

        self.sql_generator: SQLGeneratorAgent | None = None

        self.llm_config = None

        self.create_sql_generator()

    def create_sql_generator(self):
        """
        Create the SQL generator agent based on the model type and model name.
        """
        self.sql_generator = SQLGeneratorAgent(
            db_config=self.db_config, llm_config=self.llm_config, verbose=self.verbose
        )
        return self.sql_generator

    def eval(self) -> List[EvalResultItem]:
        """Evaluate the generated SQL queries against the ground truth queries."""
        # first we have a dataframe with the questions and the ground truth queries
        eval_items: List[EvalItem] = self.loader.load_eval_items(max_rows=self.num_questions)
        eval_results: List[EvalResultItem] = []

        for eval_item in eval_items:
            logger.info(f"Evaluating question: {eval_item.question}, the database is {eval_item.db_name}...")

            generated_query = self.sql_generator.generate_sql_with_agent(eval_item.question, single_line_format=True)
            result_item = EvalResultItem(
                question=eval_item.question,
                query=eval_item.golden_query,
                db_name=eval_item.db_name,
                query_category=eval_item.query_category,
                instructions=eval_item.instructions,
                generated_query=generated_query,
            )

            if not generated_query:
                logger.error(f"Failed to generate query for question: {eval_item.question}.")
                eval_results.append(result_item)
                continue

            logger.info(f"Generated query: {generated_query}.")

            logger.info("Evaluating the correctness of the generated query...")
            result_item.exec_correct = self.eval_exec_correctness(
                generated_query, eval_item.golden_query, eval_item.db_name
            )
            result_item.is_correct = result_item.exec_correct

            eval_results.append(result_item)

        self.save_output(eval_result=eval_results)

        return eval_results

    def eval_exec_correctness(self, generated_query: str, golden_query: str, db_name: str):
        """Check the execution correctness of the generated queries.

        Args:
            generated_query (str): query to be executed
            golden_query (str): golden query to be compared with
            db_name (str): the database name to be connected
        """
        if generated_query == golden_query:
            return True

        if not self.db_engine or self.db_engine.db_config.db_name != db_name:
            # if the db engine is not initialized or the db name is different, we need to reconnect
            self.get_db_engine(db_name=db_name)

        # execute the generated query
        try:
            my_result = self.query_db(generated_query, db_name)
            logger.info(f"My SQL result: {my_result}")
            golden_result = self.query_db(golden_query, db_name)
            logger.info(f"Golden SQL result: {golden_result}")
            return my_result == golden_result
        except ValueError as e:
            logger.error(f"Error in executing the query: {e}")
            return False

    def save_output(self, eval_result: List[EvalResultItem]) -> str:
        """Save the evaluation output to a JSON file.

        Args:
            eval_result (List[EvalResultItem]): the evaluation result to be saved

        Returns:
            str: path to the saved file, will connect the prefix with the current time
        """
        save_path = f"{self.output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        eval_result_df = pd.DataFrame([item.dict() for item in eval_result])
        eval_result_df.to_json(save_path, orient="records", lines=True)
        logger.info(f"Saved the evaluation result to {save_path}.")
        return save_path

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
        if self.db_engine:
            return self.db_engine

        # first call, so we need to initialize the db engine
        if not self.db_config and not self.db_type:
            raise ValueError("Database configuration is missing.")

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

    def query_db(self, query: str, db_name: str):
        """Query the database.

        Args:
            query (str): the query to be executed
            db_name (str): the database name to be connected

        Returns:
            dict: The query result
        """
        engine = self.get_db_engine(db_name=db_name)
        return engine.execute(query)


if __name__ == "__main__":
    abs_path = Path(__file__).parent / "questions_gen_postgres.csv"
    evaluator = Evaluator(
        dataset_path=abs_path,
        num_questions=1,
        model_type="azure",
        db_type="postgres",
        verbose=True,
    )
    results = evaluator.eval()
    print(results)
