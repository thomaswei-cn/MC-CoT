# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from utils.register import registry
from typing import Callable, List, Optional, Union
import json
import copy


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual_model_name', default='LLava', choices=['LLava', 'QwenVL', 'QwenVL-Max', 'DeepseekVL'])
    parser.add_argument('--language_model_name', default="GPT", choices=['GPT', 'ChatGLM', 'Deepseek', 'Qwen2'])
    parser.add_argument('--method', type=str, default='VisualOnly')
    parser.add_argument('--dataset_name', choices=['PATH-VQA', 'VQA-RAD', 'Slake'])
    parser.add_argument('--slake_path', help='Path to the slake file folder')
    parser.add_argument('--pmc_path', help='Path to the pmc file folder')
    parser.add_argument('--path_vqa_path', help='Path to the path-vqa file folder')
    parser.add_argument('--vqa_rad_path', help='Path to the vqa-rad file folder')
    parser.add_argument('--max_retries', type=int, default=5)
    parser.add_argument('--v_device', type=int, default=0)
    parser.add_argument('--l_device', type=int, default=0)
    parser.add_argument('--ff_print', type=bool, default=False)
    args = parser.parse_args()

    return args

                