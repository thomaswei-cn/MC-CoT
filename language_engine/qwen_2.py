import os

from openai import OpenAI

from utils.register import register_class
from .base_language_engine import BaseLanguageEngine
from transformers import AutoModelForCausalLM, AutoTokenizer


@register_class(alias="Engine.Qwen2")
class Qwen2Engine(BaseLanguageEngine):
    def __init__(self, device=0, model_name='qwen2-72b-instruct', temperature=0.0, seed=127):
        dashscope_api_key = os.environ.get('Dashscope_API_KEY')
        assert dashscope_api_key is not None
        qwen2_api_base = os.environ.get('Qwen2_API_BASE')
        assert qwen2_api_base is not None

        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.client = OpenAI(
            api_key=dashscope_api_key,
            base_url=qwen2_api_base)

    def get_response(self, user_input, system_input=""):
        response = None
        i = 0
        messages = [{'role': 'system', 'content': system_input},
                   {'role': 'user', 'content': user_input}]
        while i < 3:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    seed=self.seed,
                    max_tokens=1000
                )
            except Exception as e:
                print(e)
            i += 1
            if response is not None and response.choices[0].message.content is not None:
                response = response.choices[0].message.content
                break
            else:
                continue
        return response

