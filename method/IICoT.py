from tqdm import tqdm
from utils.register import register_class, registry
from .base_method import BaseMethod
from utils.output_utils import ensure_dir, format_json_out_put, filter_finished, format_output_filepath


def get_llm_guide(question, domain):
    role = ""
    if domain == "PATH-VQA":
        role = "You are a pathologist.\n"
    elif domain == "VQA-RAD":
        role = "You are a radiology expert.\n"
    elif domain == "Slake":
        role = "You are a medical expert.\n"
    prompt = (
        f"You are good at analyzing question.Here is a question.\nQuestion:{question}, however, you can't see the images. Please give a detailed guide for non-professionals on how to give the right answer.\n"
        f"Points to note:\n"
        f"1. You need to explain the features that the image may contain based on the question, and how to give the right answer from the perspective of the picture.\n"
        f"2. Remember you are teaching a rookie to read these medical images. So make sure you break down medical or biological terms into intuitive descriptions, especially terms related to image features.\n"
        f"3. You cannot give your speculation on the final answer.")
    return role + prompt


def get_rationale_with_guide(question, llm_guide):
    prompt = f"Let's work this out in a step by step way to be sure we have the right answer." \
           f"Question: {question}\n Here is a guide of the question.\nGuide:\n{llm_guide}\n\n" \
           "Points to note: 1: The analyses provided should guide you towards the correct response.\n " \
           "2: Use the knowledge you learned from the guide and give the analyses of the quesition." \
           "3: If the correct answer is already given in the guide, you just simply restate it in the output."
    return prompt


def get_final_prompt(question, rationale):
    prompt = f"Please infer a final answer about this question according to the rationale.\nQuestion: {question}\n Rationale{rationale}."
    return prompt


@register_class("IICoT")
class IICoT(BaseMethod):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.output_file_path = format_output_filepath(args.language_model_name, args.visual_model_name, args.method, args.dataset_name)
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
            guide_prompt = get_llm_guide(question, self.dataset.dataset_name)
            guide = self.l_engine.get_response(guide_prompt)
            rationale_prompt = get_rationale_with_guide(question, guide)
            rationale = self.v_engine.get_response(rationale_prompt, img, img_path)
            final_prompt = get_final_prompt(question, rationale)
            response = self.v_engine.get_response(final_prompt, img, img_path)
            if response is not None:
                format_json_out_put(question, answer, response, idx, self.output_file_path)
