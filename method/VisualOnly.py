from tqdm import tqdm
from utils.output_utils import format_json_out_put, filter_finished, ensure_dir
from utils.register import register_class, registry
from .base_method import BaseMethod
import random
random.seed(1234)


def get_prompt(question):
    prompt = f"Question: {question} \n" \
             f"Please directly answer the question based on the given image. "
    return prompt


@register_class(alias="Method.VisualOnly")
class VisualOnly(BaseMethod):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.output_file_path = args.output_file_path
        ensure_dir(self.output_file_path)
        self.shuffle = args.shuffle
        self.truncation_50 = args.truncation_50
        self.v_engine = registry.get_class(args.visual_model_name)(device=args.v_device)

    def run(self):
        todo_list = filter_finished(len(self.dataset), self.output_file_path)
        if self.shuffle:
            random.shuffle(todo_list)
        if self.truncation_50:
            todo_list = todo_list[:50]
        for idx in tqdm(todo_list):
            img, question, answer = self.dataset[idx]
            prompt = get_prompt(question)
            response = self.v_engine.get_response(prompt, img)
            format_json_out_put(question, answer, response, idx, self.output_file_path)
