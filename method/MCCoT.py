import re
from utils.output_utils import format_json_out_put, filter_finished, ensure_dir
from tqdm import tqdm
from utils.register import register_class, registry
from .base_method import BaseMethod


def parse_domain(dataset_name):
    if dataset_name == "Slake":
        domain = "slake"
    elif dataset_name == "PATH-VQA":
        domain = "path"
    elif dataset_name == "VQA-RAD":
        domain = "rad"
    else:
        domain = ""
    return domain


def match_tasks(decision):
    pattern_radiology = r"Radiology Module:\s+((?: *- .+?\n)+)|Radiology Module:\s+(.+?)(?=\n2\.)"
    pattern_anatomy = r"Anatomy Module:\s+((?: *- .+?\n)+)|Anatomy Module:\s+(.+?)(?=\n2\.)"
    pattern_pathology = r"Pathology Module:\s+((?: *- .+?\n)+)|Pathology Module:\s+(.+?)(?=\n2\.)"

    match_radiology = re.search(pattern_radiology, decision, re.DOTALL)
    match_anatomy = re.search(pattern_anatomy, decision, re.DOTALL)
    match_pathology = re.search(pattern_pathology, decision, re.DOTALL)

    return {"radiology": match_radiology, "anatomy": match_anatomy, "pathology": match_pathology}


def is_module_required(text):
    if text != "N/A" and text != "None" and text.find("Not applicable") == -1 and text.find("Not required") == -1:
        return True
    return False


def get_description_prompt(question, domain):
    prompt = (f"You're good at answering questions. Here is a question: {question}.\n"
              f"Please provide detailed descriptions of the features which you think is relative to the question.\n"
              "You should focus on the image's content, "
              "such as the color or brightness of certain areas, shapes of visible objects and their locations in the image.\n"
              "Never use common sense to make general descriptions, your discriptions must be based entirely on the image.\n"
              "Do not make any assumptions or conclusions about the image, like the subject or body part of the image and so on.")
    return prompt


def get_decision_prompt(question, description, domain):
    prompt_pt1 = (
        "You are a advanced question-answering agent equipped with 3 specialized modules to aid in analyzing and responding to questions about medical images:\n\n"
        "1. Radiology Module:\n"
        "Abilities:\n"
        "1) Determine the appropriate imaging modality (e.g., CT, MRI, Ultrasound).\n"
        "2) Identify the imaging plane (e.g., axial, sagittal, coronal).\n"
        "3) Pinpoint the position of the lesion within the image.\n"
        "4) Analyze the color/contrast characteristics on the imaging study to differentiate tissue types and abnormalities.\n"
        "When this module is required, specify your request as: 'Radiology Module: <specific task or information to extract>.'\n\n"
        "2. Anatomy Module:\n"
        "Abilities:\n"
        "1) Identify the organ or anatomical structure involved.\n"
        "2) Provide detailed information on the anatomical position and relations of the lesion within the body.\n"
        "When you need this module, specify your request as: 'Anatomy Module: <specific task or information to extract>.'\n\n"
        "3. Pathology Module:\n"
        "Abilities:\n"
        "1) Consider the number of lesions and their clinical significance.\n"
        "2) Provide a reasonable explanation for the phenomenon in combination with pathology knowledge.\n"
        "When information from this module is needed, specify your request as: \"Pathology Module: <specific task or information to extract>.\"\n\n"
    )

    prompt_pt2 = (
        "When faced with a question about an image, which will be accompanied by a description that might not cover all its details, your task is to:\n\n"
        "- Provide a rationale for your approach to answering the question, explaining how you will use the information from the image and the modules to form a comprehensive answer.\n"
        "- Assign specific tasks to each module as needed, based on their capabilities, to gather additional information essential for answering the question accurately.\n\n"
        "Your response should be structured as follows:\n\n"
        "Answer: [Rationale: Your explanation of how you plan to approach the question, including any initial insights based on the question and image description provided. Explain how the modules' input will complement this information.]\n\n")
    prompt_pt3 = ("Modules' tasks  (if applicable):\n\n"
                  "1. Radiology Module: [Clearly list in detail the tasks that need to be completed by the radiology module.]\n"
                  "2. Anatomy Module: [Clearly list in detail the tasks that need to be completed by the anatomy module.]\n"
                  "3. Pathology Module: [Clearly list in detail the tasks that need to be completed by the pathology module.]\n\n"
                  "Ensure your response adheres to this format to systematically address the question using the available modules or direct analysis as appropriate.\n"
                  f"Please refer to the prompts and examples above to help me solve the following problem:{question}\n"
                  f"Here is a description of the related medical image: {description}")

    return prompt_pt1 + prompt_pt2 + prompt_pt3


def get_guide_prompt(task, question):
    prompt = (f"Here is a question based on a medical image: {question}\n"
              f"In order to answer the question correctly, you are assigned to complete the following task: {task}\n"
              f"You cannot see the image, please use your medical knowledge to provide a guide on how to solve the task.\n"
              "Points to note:\n"
              f"1. You need to explain the features that the image may contain based on the task, and how to give the right answer from the perspective of the picture.\n"
              f"2. Remember you are teaching a rookie to read a medical image. So make sure you break down medical or biological terms into intuitive descriptions, especially terms related to image features.\n"
              f"3. You cannot give your speculation on the final answer."
              )
    return prompt


def get_mllm_answer_prompt(task, guide):
    mllm_prompt = f"Please give a detailed response to this task according to the guide.\nTask: {task}\n Guide: {guide}.\n"
    return mllm_prompt


def get_integrate_answer_prompt(question, rad, atm, pth, des):
    suplement = ""
    if rad is not None:
        suplement += f"Radiology Module: {rad}. "
    if atm is not None:
        suplement += f"Anatomy Module: {atm}. "
    if pth is not None:
        suplement += f"Pathology Module: {pth}. "
    prompt_pt1 = (
        "You are a knowledgeable and skilled information integration medical expert. Please gradually think and"
        "answer the questions based on the given questions and supplementary information.\n"
        "Please note that we not only need answers, but more importantly, we need rationales for obtaining answers.\n"
        "Please prioritize using your knowledge to answer questions.\n"
        "Furthermore, please do not rely solely on supplementary information, as the provided supplementary"
        " information may not always be effective.\n"
        "Please do not answer with uncertainty, try your best to give an answer.\n"
        f"Here is a description of the related medical image: {des}.\n")
    if rad is None and atm is None and pth is None:
        prompt = (f"{prompt_pt1}"
                  f"The expected response format is as follows: Rationale:<rationale> Answer:<answer>.\n"
                  f"Please answer the following case: Question: <{question}> ,Supplementary information: {des}")
    else:
        prompt = (f"{prompt_pt1}"
                  f"The expected response format is as follows: Rationale:<rationale> Answer:<answer>.\n"
                  f"Please answer the following case: Question: <{question}> ,Supplementary information: {suplement}.")

    return prompt


@register_class(alias="Method.MCCoT")
class MCCoT(BaseMethod):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.modules = ["radiology", "anatomy", "pathology"]
        self.domain = parse_domain(args.dataset_name)
        self.output_file_path = args.output_file_path
        ensure_dir(self.output_file_path)
        self.max_retries = args.max_retries
        self.v_engine = registry.get_class(args.visual_model_name)(device=args.v_device)
        self.l_engine = registry.get_class(args.language_model_name)(device=args.l_device)

        self.ff_print = args.ff_print

    def run(self):
        for round_count in range(self.max_retries):
            print("Start {} round of answering questions.".format(round_count + 1))
            todo_list = filter_finished(len(self.dataset), self.output_file_path)
            if not todo_list:
                print("All questions have been answered.")
                return
            for idx in tqdm(todo_list):
                mllm_answer_dict = {}
                img, question, answer = self.dataset[idx]
                description = self.v_engine.get_response(get_description_prompt(question, self.domain), img)
                if self.ff_print:
                    print(f"Description: {description}")
                decision = self.l_engine.get_response(get_decision_prompt(question, description, self.domain))
                if self.ff_print:
                    print(f"Decision: {decision}")
                if decision is None:
                    continue
                match_results = match_tasks(decision)
                for module in self.modules:
                    match = match_results[module]
                    if match:
                        if match.group(1):
                            task = match.group(1).strip()
                        else:
                            task = match.group(2).strip()
                        if is_module_required(task):
                            guide = self.l_engine.get_response(get_guide_prompt(task, question))
                            if self.ff_print:
                                print(f"Guide: {guide}")
                            mllm_answer = self.v_engine.get_response(get_mllm_answer_prompt(task, guide), img)
                            if mllm_answer.find("Answer:") != -1:
                                mllm_answer = mllm_answer[mllm_answer.find("Answer:") + len("Answer:"):].strip()
                            if mllm_answer == "":
                                mllm_answer = None
                        else:
                            mllm_answer = None
                    else:
                        task = None
                        mllm_answer = None
                    mllm_answer_dict[module] = mllm_answer
                if self.ff_print:
                    print(f"MLLM Answer Dict: {mllm_answer_dict}")
                final_answer = self.l_engine.get_response(
                    get_integrate_answer_prompt(question, mllm_answer_dict["radiology"],
                                                mllm_answer_dict["anatomy"],
                                                mllm_answer_dict["pathology"], description))
                if self.ff_print:
                    print(f"Final Answer: {final_answer}")
                format_json_out_put(question, answer, final_answer, idx, self.output_file_path)
