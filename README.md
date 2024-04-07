# Text-to-SQL-Agent

## Introduction

A text to sql agent using LLMs.

## Usage

Make sure python>=3.10

## Test
```shell
pytest --cov=text_to_sql --cov-report html
```

## Run
```shell
pip install -r requirement.dev.txt
```

create a .env file under text_to_sql/config, set below env:

```shell
# Necessary
# LLM Config
AZURE_ENDPOINT="https://endpoint.azure.example.com"
AZURE_API_KEY="xxx"

# Database config
DB_HOST="xxxx"
DB_USER="xxxx"
DB_PASSWORD="xxxx"
DB_NAME="xxxx"

# Belows are optional

DEBUG="True"
# if using gpt4 model
AZURE_GPT_4="False"

# Embedding model
AZURE_EMBEDDING_MODEL=""
HUGGING_FACE_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"

```