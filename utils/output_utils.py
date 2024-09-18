import json
import os


def format_json_out_put(question, answer, output, idx, json_filename):
    sample = {
        "id": idx,
        "question": question,
        "answer": answer,
        "pred": output
    }

    # 以追加模式打开JSONL文件并写入新数据
    with open(json_filename, 'a') as json_file:
        json_file.write(json.dumps(sample) + '\n')


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


def temp_examine(json_filename):
    temp = []
    log = []
    new_lines = None
    # 读取JSONL文件
    with open(json_filename, 'r') as json_file:
        lines = json_file.readlines()
        for line in lines:
            id = json.loads(line)['id']
            if id in temp:
                log.append(id)
            else:
                temp.append(id)
        for line2 in lines:
            id = json.loads(line2)['id']
            if id in log:
                # 删除这一行
                lines.remove(line2)
        new_lines = lines
    # 写入JSONL文件
    with open(json_filename, 'w') as json_file:
        for line in new_lines:
            json_file.write(line)
    print(log)

