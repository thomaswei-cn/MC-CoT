from tqdm import tqdm
from utils.register import register_class, registry
from .base_method import BaseMethod
from utils.output_utils import ensure_dir, format_json_out_put, filter_finished


def get_qvix_prompt_stg1(question):
    task = "Answer the following science question about an image.\n"
    query = question
    prompt_gptrewrite_v1 = f'''I require assistance in formulating a response to a central inquiry regarding a specific image: 
    {task}{query}

    The task is to create 4 preliminary questions. These questions should zero in on crucial contextual details within the image that are pertinent to addressing the main inquiry.

    Guidelines for the preliminary questions:

    Each question must be concise and easily comprehensible.
    They should concentrate on contextual visual elements present in the image.
    These questions ought to offer insights that aid in responding to the main question.
    Proposed Format:
    Preliminary Question 1: xxxx
    Preliminary Question 2: xxxx
    Preliminary Question 3: xxxx
    Preliminary Question 4: xxxx'''
    return prompt_gptrewrite_v1


def get_qvix_prompt_stg2(question, preques):
    task = "Answer the following science question about an image.\n"
    query = question
    return preques + task + query


@register_class("Qvix")
class Qvix(BaseMethod):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.output_file_path = f'./outputs/{args.language_model_name}/{args.visual_model_name}/{args.method}/{args.method}_{dataset}.jsonl'
        ensure_dir(self.output_file_path)
        self.max_retries = args.max_retries
        self.v_engine = registry.get_class(args.visual_model_name)(device=args.v_device)
        self.l_engine = registry.get_class(args.language_model_name)(device=args.l_device)

    def run(self):
        for round_count in range(self.max_retries):
            print("Start {} round of answering questions.".format(round_count + 1))
            todo_list = filter_finished(len(self.dataset), self.output_file_path)
            if not todo_list:
                print("All questions have been answered.")
                return
            self._run_list(todo_list)
        print("Max retries reached.")

    def _run_list(self, todo_list):
        for idx in tqdm(todo_list):
            img, question, answer, img_path = self.dataset[idx]
            preque_prompt = get_qvix_prompt_stg1(question)
            preque = self.l_engine.get_response(preque_prompt)
            answer_prompt = get_qvix_prompt_stg2(question, preque)
            response = self.v_engine.get_response(answer_prompt, img, img_path)
            if response is not None:
                format_json_out_put(question, answer, response, idx, self.output_file_path)
