import argparse
import json
import os


def get_total_score_with_len(json_filename):
    total_score = 0
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            lines = json_file.readlines()
            for line in lines:
                total_score += json.loads(line)['score']
        return total_score, len(lines)
    else:
        return -1, -1


class EvalShower:
    def __init__(self, args):
        self.method = args.method
        self.dataset_name = args.dataset_name
        self.v_model = args.v_model
        self.l_model = args.l_model

    def run(self):
        for m in self.method:
            print("-" * 40)
            print(f"Method: {m}")
            sum = 0
            for d in self.dataset_name:
                output_file_path = f'../outputs/eval/{self.l_model}/{self.v_model}/{m}/{m}_{d}_eval.jsonl'
                total_score, total_len = get_total_score_with_len(output_file_path)
                if total_score == -1:
                    print(f"Dataset: {d}, No Data")
                    continue
                avg = total_score / total_len
                scale_score = round((avg - 1) * 33.33333333333333, 2)
                sum += scale_score
                print(f"Dataset: {d}, Avg. Score: {scale_score}%, Total Len: {total_len}")
            print(f"Total Avg.: {round(sum / len(self.dataset_name), 2)}%")
        print("-" * 40)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', nargs='+', type=str, default=['M3'])
    parser.add_argument('--dataset_name', type=str, nargs='+', default=['Slake', 'PATH', 'RAD'],
                        choices=['Slake', 'PATH', 'RAD'])
    parser.add_argument('--v_model', choices=['qwen', 'qwen-max', 'deepseek', 'llava'], default='llava')
    parser.add_argument('--l_model', choices=['gpt3.5', 'deepseek', 'qwen2', 'chatGLM'], default='gpt3.5')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    shower = EvalShower(args)
    shower.run()
