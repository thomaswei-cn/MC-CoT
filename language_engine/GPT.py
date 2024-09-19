import os
import openai
from openai import OpenAI
from utils.register import register_class
from .base_language_engine import BaseLanguageEngine
import time


@register_class(alias="GPT")
class GPTEngine(BaseLanguageEngine):
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.0, seed=127, device=0):
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        assert openai_api_key is not None
        openai_api_base = os.environ.get('OPENAI_API_BASE')

        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed

        if openai_api_base is not None:
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base
            )
        else:
            self.client = OpenAI(
                api_key=openai_api_key,
            )

    def get_response(self, user_input, system_input=""):
        response = None
        model_name = self.model_name
        i = 0
        messages = [{"role": "system", "content": system_input},
                    {"role": "user", "content": user_input}]
        while i < 5:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    seed=self.seed
                )
                if response is not None:
                    break
            except openai.BadRequestError:
                if model_name == "gpt-3.5-turbo":
                    model_name = "gpt-3.5-turbo-16k"
                i += 1
            except openai.RateLimitError:
                time.sleep(10)
                i += 1
            except Exception as e:
                print(e)
                i += 1
                time.sleep(5)
                continue
            else:
                i += 1
        if response is not None:
            return response.choices[0].message.content
        else:
            # print("Failed to get response from GPT, messages: ", messages)
            return None
