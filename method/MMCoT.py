from tqdm import tqdm
from utils.output_utils import format_json_out_put, filter_finished, ensure_dir, format_output_filepath
from utils.register import register_class, registry
from .base_method import BaseMethod


def get_prompt_1(question):
    prompt = f"Please generate a detailed rationale about this question.\nQuestion: {question}\n.You need to focus on the rationale generation and you can't give the final answer"
    return prompt


def get_prompt_2(question, rationale):
    prompt = f"Please infer a final answer about this question according to the rationale.\nQuestion: {question}\n Rationale{rationale}.\n\n  Please respond only with the selected option's letter, like A, B, C, D, using the following format: '''Answer: [Selected Option's Letter]'''."
    return prompt


@register_class(alias="MMCoT")
class MMCoT(BaseMethod):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.output_file_path = format_output_filepath(args.language_model_name, args.visual_model_name, args.method, args.dataset_name)
        ensure_dir(self.output_file_path)
        self.v_engine = registry.get_class(args.visual_model_name)(device=args.v_device)

    def run(self):
        todo_list = filter_finished(len(self.dataset), self.output_file_path)
        for idx in tqdm(todo_list):
            img, question, answer, img_path = self.dataset[idx]
            prompt = get_prompt_1(question)
            rationale = self.v_engine.get_response(prompt, img, img_path)
            prompt_2 = get_prompt_2(question, rationale)
            response = self.v_engine.get_response(prompt_2, img, img_path)
            format_json_out_put(question, answer, response, idx, self.output_file_path)
