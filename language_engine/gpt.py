import os
from utils.register import register_class
from .base_language_engine import BaseLanguageEngine
import time
import requests

base_url = "http://preview-general-llm.api.ai.srv/api/%s/weilai8"
@register_class(alias="Engine.GPT")
class GPTEngine(BaseLanguageEngine):
    def __init__(self, model_name="gpt-4-32k-0314", temperature=0.0, seed=127, device=None):
        self.api_base = base_url % model_name
        self.temperature = temperature
        self.model_name = model_name

    def get_response(self, user_input, system_input=""):
        i = 0
        messages = [{"role": "system", "content": system_input},
                    {"role": "user", "content": user_input}]
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "messages": messages,
            "temperature": self.temperature,
            # "response_format": "json_object"
        }
        res_content = None
        res_usage = None
        while True:
            try:
                res = requests.post(self.api_base, json=data)
                # print(f"Request API successfully, response: {res.json()}")
                res_json = res.json()
                if res_json["status"] != 200:
                    # 经常出现token rate limit, 重试
                    print(f"Bad Request,, retrying...")
                    time.sleep(1.3)
                    continue

                else:
                    res_content = res_json['response']
                    res_usage = res_json['usage']
                    break
            except:
                print(f"Error: res: {res}")
                break
            # else:
            #     break
            # break

        # print(f"Request API successfully, response: {res_content}")
        return res_content
