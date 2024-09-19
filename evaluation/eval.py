import argparse
import json
import os
import re
from transformers import AutoTokenizer
from eval_recall import calculate
from openai import OpenAI
from tqdm import tqdm
import concurrent


def filter_finished(total_len, json_filename):
    finished = []
    total = range(total_len)
    if os.path.exists(json_filename):
        # 读取JSONL文件
        with open(json_filename, 'r') as json_file:
            lines = json_file.readlines()
            for line in lines:
                finished.append(json.loads(line)['id'])
    return list(set(total) - set(finished))


def ensure_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_eval_acc_prompt(question, answer, pred, dataset_name):
    sys = (
        f"You're an expert in evaluating {dataset_name} dataset answers. Your task is to evaluate the predict answer generated by a Multimodal model."
        f"Every generated answer will be paired with a correct answer, consider the correct answer as a gold standard.You need to rate the generated answer from 1 to 4 according to the following criteria:\n"
        f"1: No answer is given, or the generated answer is refering totally different thing, imaging type or body part from the correct answer, or the generated answer is simply restating the information the question has given.\n"
        f"2: The generated answer includes multiple guesses, among which there is the correct answer or part of the correct answer, though wrong answers may also be mentioned.\n"
        f"3: The generated answer only contains part of the correct answer or just an aspect of the correct answer, with no wrong answers included.\n"
        f"4: The generated answer is identical to the correct answer.\n"
        f"Remember, the question is just for reference, when rating the answer, do not consider the relation between the generated answer and the question or the context.\n")

    user = (
        f"And since you can not see the image, please consider that the image only show exactly what the correct answer tells you.\n"
        f"Now, here is a question based on some medical images: {question}, and the correct answer is {answer}. \n"
        f"Please follow the instructions above and evaluate the generated answer: {pred}\n"
        f"Your output should include your reason and be in the exact same format as '''Result:[Your rating]'''")
    return sys, user


def read_jsonl_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_eval_output(question, answer, output, idx, score, json_filename):
    sample = {
        "id": idx,
        "question": question,
        "answer": answer,
        "pred": output,
        "score": score
    }

    # 以追加模式打开JSONL文件并写入新数据
    with open(json_filename, 'a') as json_file:
        json_file.write(json.dumps(sample) + '\n')


def get_gpt_response(sys, user, client):
    response = None
    i = 0
    message = [{'role': 'system', 'content': sys},
               {'role': 'user', 'content': user}]
    while i < 3:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=message,
            temperature=0,
            seed=127,
            max_tokens=1000)
        i += 1
        if response is not None and response.choices[0].message.content is not None:
            response = response.choices[0].message.content
            # print(response)
            if response.rfind("Result:") == -1:
                # print("No result found")
                response = None
                continue
            else:
                match = re.search(r"Result:\s*(\d)", response)
                # 如果找到了匹配
                if match:
                    response = match.group(1)
                    # print(response)
                    return response
                else:
                    # print("No result found2")
                    response = None
                    continue
        else:
            response = -100
            continue
    return response


def parse_pred(predicted_answer):
    if predicted_answer is None:
        raise ValueError("None prediction")
    if predicted_answer.find('Answer: ') != -1:
        pred = predicted_answer[predicted_answer.find('Answer: ') + len('Answer: '):]
    elif predicted_answer.find('Option: ') != -1:
        pred = predicted_answer[predicted_answer.find('Option: ') + len('Option: '):]
    else:
        pred = predicted_answer
    return pred


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', nargs='+', type=str, default=['MCCoT'])
    parser.add_argument('--dataset_name', type=str, nargs='+', default=['Slake', 'PATH-VQA', 'VQA-RAD'],
                        choices=['Slake', 'PATH-VQA', 'VQA-RAD'])
    parser.add_argument('--v_model', nargs='+', choices=['QwenVL', 'LLava'], default=['LLava'])
    parser.add_argument('--l_model', nargs='+', choices=['GPT', 'Deepseek', 'ChatGLM', 'Qwen2'], default=['GPT'])
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--openai_api_base', type=str, default=None)
    parser.add_argument('--mode', type=str, default='acc', choices=['acc', 'recall'])
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()
    return args


class Evaluator:
    def __init__(self, args):
        self.method = args.method
        self.dataset_name = args.dataset_name
        self.v_model = args.v_model
        self.l_model = args.l_model
        self.parallel = args.parallel
        self.max_workers = args.max_workers
        self.mode = args.mode

        if self.mode == 'acc':
            deepseek_api_key = os.environ.get('Deepseek_API_KEY')
            assert deepseek_api_key is not None
            openai_api_base = os.environ.get('Deepseek_API_BASE')

            if openai_api_base is not None:
                self.client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url=openai_api_base
                )
            else:
                self.client = OpenAI(
                    api_key=deepseek_api_key,
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def run(self):
        for l in self.l_model:
            print(f"Language Model: {l}")
            for v in self.v_model:
                print(f"Visual Model: {v}")
                for m in self.method:
                    print(f"Method: {m}")
                    if self.mode == "recall":
                        sum_recall = 0
                    for d in self.dataset_name:
                        print(f"Dataset: {d}")
                        file_path = f'../outputs/{l}/{v}/{m}/{m}_{d}.jsonl'
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"File not found: {file_path}")
                        if self.mode == 'acc':
                            self.output_file_path = f'../outputs/eval/{l}/{v}/{m}/{m}_{d}_eval.jsonl'
                            ensure_dir(self.output_file_path)
                        data = read_jsonl_file(file_path)
                        if self.mode == 'acc':
                            todo_list = filter_finished(len(data), self.output_file_path)
                            if self.parallel:
                                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                                    # 使用 map 来简化提交任务和获取结果的过程
                                    futures = [executor.submit(self._eval_one, data[idx], d) for idx
                                               in todo_list]
                                    for _ in tqdm(concurrent.futures.as_completed(futures),
                                                  total=len(todo_list)):
                                        pass
                            else:
                                for idx in tqdm(todo_list):
                                    item = data[idx]
                                    self._eval_one(item, d)
                        else:
                            total_recall = 0
                            for item in data:
                                recall = calculate(item['answer'], parse_pred(item['pred']), self.tokenizer)
                                total_recall += recall
                            print(f"Recall: {round((total_recall / len(data)) * 100, 2)}%")
                            sum_recall += total_recall / len(data)
                    print(f"Average Recall: {round((sum_recall / len(self.dataset_name)) * 100, 2)}%")
                    print("-" * 50)

    def _eval_one(self, item, dataset_name):
        sys, user = get_eval_acc_prompt(item['question'], item['answer'], item['pred'], dataset_name)
        response = get_gpt_response(sys, user, self.client)
        if response is None:
            score = 1
        else:
            if response == -100:
                raise ValueError("token limit reached")
            score = int(response)
        format_eval_output(item['question'], item['answer'], item['pred'], item['id'], score,
                           self.output_file_path)


if __name__ == '__main__':
    args = get_args()
    evaluator = Evaluator(args)
    evaluator.run()
