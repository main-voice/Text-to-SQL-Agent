import unittest

from text_to_sql.llm.prompts import AgentPlanPromptBuilder


class TestAgentPlanPromptBuilder(unittest.TestCase):
    """
    Class to test AgentPlanPromptBuilder
    """

    def test_build_plan(self):
        db_type = "postgres"
        builder = AgentPlanPromptBuilder(db_type=db_type)
        test_instructions = "This is a test instructions"
        plan_template1 = builder.build_plan_template(instructions=test_instructions, add_current_time=True)
        plan1 = plan_template1.format(instructions=test_instructions)
        print(plan1)
        self.assertTrue("instructions" in plan1)
        self.assertTrue("time" in plan1)
        self.assertTrue(db_type in plan1)

        plan_template2 = builder.build_plan_template(instructions=None, add_current_time=False)
        plan2 = plan_template2.format()
        print(plan2)
        self.assertFalse("instructions" in plan2)
        self.assertFalse("time" in plan2)
        self.assertTrue(db_type in plan2)
