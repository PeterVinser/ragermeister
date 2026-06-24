from typing import Type, TypeVar
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from solution.models.message import Message
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class LLM:
    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model = model
        self.client = AzureOpenAI()
        self.input_token_count = 0
        self.output_token_count = 0

    def get_response(self, system_prompt: str, messages: list[Message]) -> str | None:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages.extend([message.model_dump() for message in messages])

        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages, # type: ignore
            verbosity="low",
            reasoning_effort="none"
        )

        if response.usage:
            self.input_token_count += response.usage.prompt_tokens
            self.output_token_count += response.usage.completion_tokens

        response_message = response.choices[0].message.content
        
        return response_message
    
    def get_structured_response(self, system_prompt: str, messages: list[Message], model_class: Type[T]) -> T:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages.extend([message.model_dump() for message in messages])

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=all_messages, # type: ignore
            response_format=model_class,
            reasoning_effort="none",
            verbosity="low"
        )
        
        if response.usage:
            self.input_token_count += response.usage.prompt_tokens
            self.output_token_count += response.usage.completion_tokens

        return response.choices[0].message.parsed