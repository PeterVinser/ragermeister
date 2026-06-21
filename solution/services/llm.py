from openai import AzureOpenAI
from solution.models.message import Message

class LLM:
    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model = model
        self.client = AzureOpenAI()
        self.input_token_count = 0
        self.output_token_count = 0

    async def get_response(self, system_prompt: str, messages: list[Message]) -> str:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages.extend([message.model_dump() for message in messages])
        
        kwargs = {}
        
        if self.model.startswith("gpt-5"):
            kwargs["reasoning_effort"] = "none"

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=all_messages, # type: ignore
            tool_choice="required" if include_tools else None, # type: ignore
            verbosity="low" if self.model.startswith("gpt-5") else "medium",
            **kwargs
        )

        if response.usage:
            self.input_token_count += response.usage.prompt_tokens
            self.output_token_count += response.usage.completion_tokens

        response_message = response.choices[0].message
        
        return response_message