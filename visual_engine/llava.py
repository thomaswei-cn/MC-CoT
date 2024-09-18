import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from utils.register import register_class
from .base_visual_engine import BaseVisualEngine


@register_class(alias="Engine.LLava")
class LLavaEngine(BaseVisualEngine):
    def __init__(self, temperature=0.0, max_tokens=2500, device=0):
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.float16,
                                                                   low_cpu_mem_usage=True)
        self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_response(self, user_input, image):
        prompt = "USER: <image>\n" + user_input + "\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.device, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        output = self.processor.decode(output[0][2:], skip_special_tokens=True)
        output = output[output.rfind("ASSISTANT: ") + len("ASSISTANT: "):]
        return output
