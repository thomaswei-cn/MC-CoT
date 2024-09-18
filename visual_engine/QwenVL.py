import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.register import register_class
from .base_visual_engine import BaseVisualEngine


@register_class(alias="QwenVL")
class QwenVLEngine(BaseVisualEngine):
    def __init__(self, temperature=0.0, max_tokens=2500, device=0):
        torch.manual_seed(1234)
        device_map = "cuda:" + str(device)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device_map,
                                                          trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    def get_response(self, user_input, image, image_path):
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': user_input},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response
