import re
from tqdm import tqdm
from utils.output_utils import format_json_out_put, filter_finished, ensure_dir
from utils.register import register_class, registry
from .base_method import BaseMethod


def get_prompt_1(question):
    sys = "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in choosing the right answer to the question. Note that insufficient information to answer questions is common."
    user = f"Given the context and questions, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions based on context and the questions. Then with the aim of helping humans answer the original question, try to answer the sub-questions. The expected answering form is as follows:\nSub-questions:\n 1. <sub-question 1>\n2. <sub-question 2>\n...\nSub-answers:\n1. <sub-answer 1> or 'Uncertain'\n2. <sub-answer 2> or 'Uncertain'\n...\nAnswer: <Your Answer> or 'Uncertain'\n\nFor a question, assume that you do not have any information about the picture, but try to answer the sub-questions and prioritize whether your general knowledge can answer it, and then consider whether the context can help. If sub-questions can be answered, then answer in as short a sentence as possible. If sub-questions cannot be determined without information in images, please formulate corresponding sub-answer into \"Uncertain\". \nOnly use \"Uncertain\" as an answer if it appears in the sub-answers. All answers are expected as concise as possible. \nHere is an attempt:\nContext: N/A \nHas An Image: yes\nQuestion: {question}\n"
    return sys, user


def get_prompt_2(question, preliminary_knowledge):
    sys = "You are a helpful, highly intelligent teacher. You will not only do your best to guide humans to the correct answer, but you will also give the rationales as a reference. "
    user = f"Given the context, questions, options, preliminary knowledge, think step by step and answer the questions. Please note that we need not only the answer, but more importantly the rationales of getting the answer. The expected answering form is as follows:\nRationale: <rationale>\nOption: <A, B, C, or D>\n\nPlease note that the preliminary knowledge given may not always be valid. Please select valid information to form the rationale and choose the relatively correct option as your answer. \nHere is an attempt:\nContext: N/A \nHas An Image: yes\nQuestion: {question}\nPreliminary knowledge: \n{preliminary_knowledge}"
    return sys, user


@register_class(alias="Method.DDCoT")
class DDCoT(BaseMethod):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.output_file_path = args.output_file_path
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
            for idx in tqdm(todo_list):
                img, question, answer = self.dataset[idx]
                sys1, user1 = get_prompt_1(question)
                response = self.l_engine.get_response(user1, sys1)
                if response is None:
                    continue
                sub_question_pattern = re.compile(r"Sub-questions:(.*?)Sub-answers:", re.DOTALL)
                sub_answer_pattern = re.compile(r"Sub-answers:(.*?)Answer:", re.DOTALL)
                sub_questions_match = sub_question_pattern.search(response)
                sub_answers_match = sub_answer_pattern.search(response)
                sub_questions = None
                sub_answers = None
                flag = 0
                if sub_questions_match is not None:
                    sub_questions = [re.sub(r'^\d+\.\s*', '', sub.strip()) for sub in
                                     sub_questions_match.group(1).split('\n')
                                     if sub.strip()]
                else:
                    flag = 1
                if sub_answers_match is not None:
                    sub_answers = [re.sub(r'^\d+\.\s*', '', sub.strip()) for sub in
                                   sub_answers_match.group(1).split('\n') if
                                   sub.strip()]
                else:
                    flag = 1
                keywords = ["uncertain", "Uncertain", "insufficient", "Insufficient", "cannot be determined",
                            "not provide",
                            "not possible"]
                preliminary_knowledge = ""
                reform_sub_q = ""
                if flag == 0:
                    for sub_q, sub_a in zip(sub_questions, sub_answers):
                        reform_sub_q = f"Question: {sub_q} Answer:"
                        if any(keyword in sub_a for keyword in keywords):
                            text = self.v_engine.get_response(reform_sub_q, img)
                            if text is None:
                                text = " "
                            preliminary_knowledge += reform_sub_q + text + "\n"
                        else:
                            preliminary_knowledge += reform_sub_q + sub_a + "\n"
                else:
                    preliminary_knowledge = response
                sys2, user2 = get_prompt_2(question, preliminary_knowledge)
                response = self.l_engine.get_response(user2, sys2)
                format_json_out_put(question, answer, response, idx, self.output_file_path)
