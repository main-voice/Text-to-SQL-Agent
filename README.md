# Text-to-SQL-Agent

## Introduction

A text-to-sql agent using LLMs. You can connect to your MySQL or Postgres Database and propose any related questions. The agent will respond with the SQL query.

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

## Evaluate

Currectly, the evaluation is based on the [defog-data](https://github.com/defog-ai/defog-data/tree/main), which is based off the schema from the Spider, but with a new set of hand-selected questions and queries grouped by query category. So **only Postgres database is supported**.

If you want to evaluate the agent, you need to download the Spider dataset and setup the Postgres database. Please refer to this repo to start your postgres instance: [sql-eval](https://github.com/defog-ai/sql-eval?tab=readme-ov-file#start-postgres-instance).

## Test Code

```shell
pytest --cov=text_to_sql --cov-report html
```
