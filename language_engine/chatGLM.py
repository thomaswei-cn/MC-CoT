import os

import torch
from openai import OpenAI

from utils.register import register_class
from .base_language_engine import BaseLanguageEngine
from transformers import AutoModelForCausalLM, AutoTokenizer


@register_class(alias="Engine.ChatGLM")
class ChatGLMEngine(BaseLanguageEngine):
    def __init__(self, device=0, model_name="THUDM/glm-4-9b-chat", temperature=0.0, seed=127):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(f"cuda:{self.device}").eval()
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.gen_kwargs = {"max_length": 5000, "do_sample": False}

    def get_response(self, user_input, system_input=""):
        inputs = self.tokenizer.apply_chat_template([{"role": "system", "content": system_input}
                                                        , {"role": "user", "content": user_input}],
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    return_dict=True
                                                    )
        inputs = inputs.to(f"cuda:{self.device}")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
