class GenericMonologueTools:
    def __init__(self, question_budget: int = 2):
        self.question_budget = question_budget
        self.question_budget_tracker = self.question_budget

    def _execute_raise_a_question(self, tool_input_query: str) -> str:
        if self.question_budget_tracker > 0:
            self.question_budget_tracker -= 1
            return f"You have chosen ask yourself: {tool_input_query}. What can you answer to yourself?"
        else:
            return "You ran out of questions. You must provide an answer now."

    def _execute_critique_the_answer(self, tool_input_query: str) -> str:
        return f"You have chosen to critique your reasoning: '{tool_input_query}'. Now, provide your critical assessment."

    def _execute_improve_based_on_critique(self, tool_input_query: str) -> str:
        return f"You have decided to improve your answer based on the critique: '{tool_input_query}'. Now, provide the improved thought or action."

    def get_tools(self):
        return {
            "raise_a_question": self._execute_raise_a_question,
            "critique_the_answer": self._execute_critique_the_answer,
            "improve_based_on_critique": self._execute_improve_based_on_critique,
        }
