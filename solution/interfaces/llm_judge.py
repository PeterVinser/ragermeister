from abc import ABC, abstractmethod


class LLMJudgeInterface(ABC):
    @abstractmethod
    def judge_raw(self, system_prompt: str, user_content: str) -> str:
        """Call the LLM with the given prompts and return the raw text response."""
