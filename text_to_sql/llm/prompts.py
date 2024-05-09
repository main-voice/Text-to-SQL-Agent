"""Store all prompt templates used in agent here"""

from typing import Callable, Dict, Optional, Tuple

from langchain.prompts import PromptTemplate


class AgentPlanPromptBuilder:
    """
    Class to build prompt templates for different agent plans
    """

    def __init__(self, db_type: str) -> None:
        self.db_type = db_type

        # A plan is Dict of steps, each step is a tuple of function and a boolean value
        self.steps: Dict[Tuple[Callable, bool]] = {
            "get_instructions_prompt": (self._get_instructions_prompt, False),
            "get_relevant_tables_prompt": (self._get_relevant_tables_prompt, True),
            "get_relevant_tables_schema_prompt": (self._get_relevant_tables_schema_prompt, True),
            "get_relevant_columns_info_prompt": (self._get_relevant_columns_info_prompt, True),
            "get_current_time_prompt": (self._get_current_time_prompt, False),
            "get_write_and_validate_prompt": (self._get_write_and_validate_prompt, True),
        }

    def build_plan_template(
        self,
        instructions: Optional[str] = None,
        add_current_time: bool = True,
    ) -> PromptTemplate:
        """
        Build the plan based on the steps and parameters

        Args:
            instructions (str): Instructions to be followed when writing SQL. Default is None
            add_current_time (bool): Whether to add the step to get current time. Default is True

        Returns:
            str: The plan prompt
        """
        # store the intermediate prompts for each step
        prompts_parts = []
        step_num = 1

        if instructions:
            # mark the get instruction step as True, meaning it will be included in the plan
            self.steps["get_instructions_prompt"] = (self._get_instructions_prompt, True)
        else:
            # mark the get instruction step as False, meaning it will not be included in the plan
            self.steps["get_instructions_prompt"] = (self._get_instructions_prompt, False)

        if add_current_time:
            # mark the get current time step as True, meaning it will be included in the plan
            self.steps["get_current_time_prompt"] = (self._get_current_time_prompt, True)
        else:
            # mark the get current time step as False, meaning it will not be included in the plan
            self.steps["get_current_time_prompt"] = (self._get_current_time_prompt, False)

        for step_func, is_include in self.steps.values():
            if is_include:
                prompts_parts.append(step_func(step_num))
                step_num += 1

        # add additional notes
        prompts_parts.append(self._get_additional_notes_prompt())

        return PromptTemplate(
            input_variables=["instructions"] if instructions else [],
            template="\n".join(prompts_parts),
        )

    def _get_relevant_tables_prompt(self, step_num: int) -> str:
        """
        Return the prompt for getting relevant tables

        """
        return f"{step_num}) Use the DatabaseTablesWithRelevanceScores tool to find relevant tables for the user query."

    def _get_relevant_tables_schema_prompt(self, step_num: int) -> str:
        """
        Return the prompt for getting relevant tables schema

        """
        return f"{step_num}) Use the DatabaseRelevantTablesSchema tool to get the schema of \
possibly relevant tables, and identify the possibly relevant columns."

    def _get_relevant_columns_info_prompt(self, step_num: int) -> str:
        """
        Return the prompt for getting relevant columns information

        """
        return f"{step_num}) Use the DatabaseRelevantColumnsInformation tool to gather more information \
about the possibly relevant columns, filtering them to find the relevant ones."

    def _get_current_time_prompt(self, step_num: int) -> str:
        return f"{step_num}. (OPTIONAL) Use the CurrentTimeTool to get the current time if \
the user question is related to time or date."

    def _get_write_and_validate_prompt(self, step_num: int) -> str:
        return f"{step_num}) Based on the collected information, write a {self.db_type} SQL query and use the \
ValidateQueryCorrectness tool to execute it on database to check if the query is correct."

    def _get_instructions_prompt(self, step_num: int) -> str:
        """
        Return the prompt for instructions
        """
        return f"{step_num}) Always follow the below instructions when writing SQL, " + "instructions: {instructions}"

    def _get_additional_notes_prompt(self) -> str:
        """
        Return some additional notes to remember when writing SQL
        """
        return """\n## Other points you need to remember:
point 1: Always execute the SQL query by calling ValidateQueryCorrectness tool to make sure the results are correct.
point 2: If the SQL query is wrong after calling ValidateQueryCorrectness tool, rewrite the SQL query and check it again
"""
