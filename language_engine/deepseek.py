import os
from openai import OpenAI
from utils.register import register_class
from .base_language_engine import BaseLanguageEngine



@register_class(alias="Engine.Deepseek")
class DeepseekEngine(BaseLanguageEngine):
    def __init__(self, device=0, model_name="deepseek-chat", temperature=0.0, seed=127):
        deepseek_api_key = os.environ.get('Deepseek_API_KEY')
        assert deepseek_api_key is not None
        deepseek_api_base = os.environ.get('Deepseek_API_BASE')
        assert deepseek_api_base is not None

        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.client = OpenAI(
            api_key=deepseek_api_key,
            base_url=deepseek_api_base)

    def get_response(self, user_input, system_input=""):
        response = None
        i = 0
        message = [{'role': 'system', 'content': system_input},
                   {'role': 'user', 'content': user_input}]
        while i < 3:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message,
                temperature=0,
                seed=127,
                max_tokens=1000)
            i += 1
            if response is not None and response.choices[0].message.content is not None:
                response = response.choices[0].message.content
                # print(response)
                break
            else:
                continue
        return response
