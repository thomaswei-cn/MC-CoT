import os.path
import pandas as pd
import random
from PIL import Image

random.seed(127)
random_pmc = random.sample(range(0, 50000), 1000)


class DatasetLoader:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        if self.dataset_name == "VQA-RAD":
            self.path = args.vqa_rad_path
        elif self.dataset_name == "Slake":
            self.path = args.slake_path
        elif self.dataset_name == "PATH-VQA":
            self.path = args.path_vqa_path
        else:
            raise Exception("Dataset not supported")
        self.data = self._load_dataset(self.dataset_name)

    def _load_dataset(self, dataset_name=None):
        if dataset_name in ["PATH-VQA", "VQA-RAD", "Slake"]:
            if dataset_name == "VQA-RAD":
                file_path = os.path.join(self.path, "VQA_RAD_open.json")
                df = pd.read_json(file_path)
            elif dataset_name == "Slake":
                file_path = os.path.join(self.path, "Slake_test_open.json")
                df = pd.read_json(file_path)
            else:
                file_path = os.path.join(self.path, "PATH-VQA_test_open.json")
                df = pd.read_json(file_path)
            return df
        else:
            raise Exception("Dataset not supported")

    def _parse_image_path(self, row_dict):
        if self.dataset_name == "VQA-RAD":
            return os.path.join(self.path, "VQA_RAD Image Folder", row_dict['image_name'])
        elif self.dataset_name == "Slake":
            return os.path.join(self.path, "imgs", row_dict['image_name'])
        elif self.dataset_name == "PATH-VQA":
            return os.path.join(self.path, "pvqa", "images", "test", row_dict['image_name'] + '.jpg')
        else:
            raise Exception("Dataset not supported")

    #  return (image, question, answer)
    def __getitem__(self, idx):
        if 0 <= idx < len(self.data):
            row = self.data.iloc[idx]
            row_dict = row.to_dict()
            img_path = self._parse_image_path(row_dict)
            img = Image.open(img_path)
            question = row_dict['question']
            answer = row_dict['answer']
            data = (img, question, answer)
            return data
        else:
            raise Exception("Dataset index out of range")

    def __len__(self):
        return len(self.data)
