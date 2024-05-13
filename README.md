# Text-to-SQL-Agent

## Introduction

A text-to-sql agent using LLMs. You can connect to your MySQL or Postgres Database and propose any related questions. The agent will respond with the SQL query.

Aim: Generate SQL queries from natural language questions using any database and LLMs.

The system support three kinds of Agent:

1. `SimpleSQLGeneratorAgent`: A simple agent that directly send `prompt + tables metadata (table name + column names + comments) + question` to LLM and get the SQL query.
2. `LangchainSQLGeneratorAgent`: A more complex agent that uses the Langchain `create_sql_query_chain` chain to generate the SQL query. It will send `prompt + tables schema + sample row data` to the LLM to generate the SQL query.
3. `SQLGeneratorAgent`: Our custom MRKL agent based on `ZeroShotAgent` in Langchain, it will use several tools to interact with database to collect necessary information and generate the SQL query.

## Usage

Make sure python>=3.10

## Environment Setup

### python env:

```shell
python -m pip install virtualenv
virtualenv venv
[windows] venv/Script/activate
[MacOS / Linux] venv/bin/activate

pip install -r requirements.dev.txt
```

### Configuration

Copy the `.env.example` file under text_to_sql/config and rename it to `.env`, set necessary configurations.

## Run

```shell
python -m text_to_sql/main.py

```

This will start a gradio website, but **only MySQL database is supported for now**.

## Evaluation

Currectly, the evaluation is based on the [defog-data](https://github.com/defog-ai/defog-data/tree/main), which is based off the schema from the Spider, but with a new set of hand-selected questions and queries grouped by query category. So **only Postgres database is supported**.

### Run eval

- Env set up: If you want to evaluate the agent, you need to download the Spider dataset and setup the Postgres database. Please refer to this repo to start your postgres instance: [sql-eval](https://github.com/defog-ai/sql-eval?tab=readme-ov-file#start-postgres-instance).

- If you are using VSCode, you can use the `Run` button to start the evaluation. I have already set up some predefined configurations in `.vscode/launch.json`.

- (Cache) The system support evaluation using previous result for cost and time saving.

- Or you can manually run the evaluation script:
  ```shell
  python -m text_to_sql/eval/evaluator.py --model_type '' --model '' --eval_method '' --num_questions '' --eval_type '' --pre_eval_result_file ''
  ```

### Visualize the evaluation result

You can use the `visualize.py` script to visualize the evaluation result, including the accuracy, cost, and time.

If you are using VSCode, you can navigate the `visualize.py` and use the `Run` button to start the visualization. I have already set up some predefined configurations in `.vscode/launch.json`.

The result will be saved in the `visualization_results` folder.

````shell
## Test Code

```shell
pytest --cov=text_to_sql --cov-report html
````
