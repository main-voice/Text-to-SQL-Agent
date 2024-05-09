"""
Store all prompt templates here
"""

from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain_core.prompts import PromptTemplate

# This is a simple prompt to generate the SQL query
SIMPLE_PROMPT = """
### ROLE
You are an assistant to interact with a SQL database to generate a correct SQL query for the given question.

### TASK
1. If instructions are provided, follow them when writing the SQL query.
2. Given an input question, generate a correct {dialect} SQL query.

### DATABASE METADATA
The generated query will run on a database with the following metadata:
{database_metadata}

### USER INPUT QUESTION
The user input question is: "{user_input}"

### Instructions
1. Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that\
s do not exist.
2. Do not add any explanations or comments for the SQL query.
3. Other instructions are: "{instructions}"

### ANSWER
Present the generated query within ```sql and ``` tags, for example:
```sql
select name, role from employees where department = 'software engineering';
```
"""


# Describe general role and task for the SQL agent, need to provide plan for the agent
SQL_AGENT_PREFIX = """
## ROLE
You are an agent designed to interact with a database to generate correct {db_type} SQL statement for a given question.

## TASK
Your task is to understand user question, interact with database using tools I will provided to get information \
you need, and follow the plan below,
and accurately generate the corresponding SQL statements. Return the SQL statement between ```sql and ``` tags.
Using `current_date()` or `current_datetime()` in SQL queries is not allowed, use CurrentTimeTool tool to \
get the exact time of the query execution if needed.

Here is the plan you need to follow step by step:
{plan}
"""

# The plan for the SQL agent, mainly describe when to use which tool
SIMPLE_PLAN = """
1. Use the DatabaseTablesWithRelevanceScores tool to get possible relevant tables for the user query.
2. Use the DatabaseRelevantTablesSchema tool to get the schema of the relevant tables, and try your best to \
identify those potential relevant columns related to user posed question.
3. Use the DatabaseRelevantColumnsInformation tool to get more information for the potentially relevant columns. \
And identify those relevant columns.
4. (OPTIONAL) Use the CurrentTimeTool to get the current time if the user question is related to time or date.
5. Generate the SQL query based on the user input and the database metadata from tools.
"""

PLAN_WITH_INSTUCTIONS = """
1. Always follow the instructions if provided when writing SQL, which are {instructions}.\
2. Use the DatabaseTablesWithRelevanceScores tool to get possible relevant tables for the user query.
3. Use the DatabaseRelevantTablesSchema tool to get the schema of the relevant tables, and try your best to \
identify those potential relevant columns related to user posed question.
4. Use the DatabaseRelevantColumnsInformation tool to get more information for the potentially relevant columns. \
And identify those relevant columns.
5. (OPTIONAL) Use the CurrentTimeTool to get the current time if the user question is related to time or date.
5. Based on the collected information, write a {db_type} SQL query directly in the next step without using any tool.\

"""

PLAN_WITH_INSTUCTIONS_AND_VALIDATION = """
1. Always follow the instructions if provided when writing SQL, which are {instructions}.\
2. Use the DatabaseTablesWithRelevanceScores tool to get possible relevant tables for the user query.
3. Use the DatabaseRelevantTablesSchema tool to get the schema of the relevant tables, and try your best to \
identify those potential relevant columns related to user posed question.
4. Use the DatabaseRelevantColumnsInformation tool to get more information for the potentially relevant columns. \
And identify those relevant columns.
5. (OPTIONAL) Use the CurrentTimeTool to get the current time if the user question is related to time or date.
6. Based on the collected information, write a {db_type} SQL query directly in the next step without using any tool.\
And always use the ValidateQueryCorrectness tool to execute it on real database to check if the query is correct.

# Other points you need to remember:
1. You should always execute the SQL query by calling ValidateQueryCorrectness tool to make sure the results are correct
2. If the SQL query is wrong after calling ValidateQueryCorrectness tool, rewrite the SQL query and check it again.
"""

PLAN_WITH_VALIDATION = """
1. Use the DatabaseTablesWithRelevanceScores tool to get possible relevant tables for the user query.
2. Use the DatabaseRelevantTablesSchema tool to get the schema of the relevant tables, and try your best to \
identify those potential relevant columns related to user posed question.
3. Use the DatabaseRelevantColumnsInformation tool to get more information for the potentially relevant columns. \
And identify those relevant columns.
4. (OPTIONAL) Use the CurrentTimeTool to get the current time if the user question is related to time or date.
5. Based on the collected information, write a {db_type} SQL query directly in the next step without using any tool.\
And always use the ValidateQueryCorrectness tool to execute it on real database to check if the query is correct.

# Other points you need to remember:
1. You should always execute the SQL query by calling ValidateQueryCorrectness tool to make sure the results are correct
2. If the SQL query is wrong after calling ValidateQueryCorrectness tool, rewrite the SQL query and check it again.
"""

# the format instructions for the SQL agent, need to provide the tool names
# essential because the agent we used is a MRKL agent (MRKL for Modular Reasoning, Knowledge and Language)
FORMAT_INSTRUCTIONS = """Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer (a valid SQL query)
Final Answer: the final answer to the original input question"""

# The suffix for the SQL agent, need to provide the input question
SQL_AGENT_SUFFIX = """Begin!

Question: {input}
Thought: I should find the relevant tables with the user input question.
{agent_scratchpad}"""


# The prompt is for translator helper
TRANSLATOR_PROMPT = """
Your primary role is to serve as a translator. Your task is to accurately translate the given text from chinese to \
english while maintaining the original meaning and context. \
Please ensure that your translations are clear, concise, and faithful to the source material. \
The input chinese: {input}
"""

# For parsing error from llm agent
ERROR_PARSING_MESSAGE = """
ERROR: Parsing error, you should only use tools or return the final answer. You are a ReAct agent, \
you should not return any other format.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, one of the tools
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If there is a consistent parsing error, please return "I don't know" as your final answer.
If you know the final answer and do not need to use any tools, directly return the final answer in this format:
Final Answer: <your final answer>.
"""

_instructions_prompt_for_langchain = """
\nFormat instruction: Wrapper your find answer in ```sql and ``` tags to \
        return the SQL statement.\
        Other instructions: {instructions}

"""
# add instructions to the langchain prompt
LANGCHAIN_POSTGRES_PROMPT = PromptTemplate(
    input_variables=["input", "top_k", "table_info", "instructions"],
    template=_postgres_prompt + _instructions_prompt_for_langchain + PROMPT_SUFFIX,
)
