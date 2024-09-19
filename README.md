# MC-CoT
This repository is the official implementation of our paper **MC-CoT: A Modular Collaborative CoT Framework for Zero-shot Medical-VQA with LLM and MLLM Integration**
![MC-CoT.jpg](src%2FMC-CoT.jpg)
# Installation
First, please use python 3.10 and create a new conda environment.
```bash
conda create -n mc-cot python=3.10
conda activate mc-cot
```
Then, you should visit the official website of PyTorch to install the correct version of PyTorch according to your system.
```
https://pytorch.org/get-started/locally/
```
Lastly, you can install the required packages by running the following command.
```bash
pip install -r requirements.txt
```
# Dataset Preparation
## PATH-VQA
Please download PATH-VQA from the official website, unzip it, and move the `PATH-VQA_test_open.json` in the `./dataset/PATH-VQA/` folder into the unzipped folder.
## SLAKE
Please download SLAKE from the official website, unzip it, and move the `Slake_test_open.json` in the `./dataset/Slake/` folder into the unzipped folder.
## VQA-RAD
Please download VQA-RAD from the official website, unzip it, and move the `VQA-RAD_test_open.json` in the `./dataset/VQA-RAD/` folder into the unzipped folder.

**Note:** In this repository, `your_path_to_{Dataset}_dir` refers to the path to the unzipped folder.
# Implementation
We have provided implementations of MCCoT as well as various CoT frameworks, along with code for calling 4 types of LLMs and 2 types of MLLMs. 

After configuring the required environment variables, such as `OPENAI_API_KEY`, you can execute MCCoT using GPT-3.5 and LLava-v1.5-7B on the SLAKE dataset with the following command:
```bash
python run.py --method MCCoT \
      --language_model_name GPT \
      --visual_model_name LLava \
      --dataset_name Slake \
      --slake_path /your_path_to_slake_dir
```
Other example scripts can be found in the `./scripts/run/` directory.

**Attention:** The output format is strictly adhere to the following: `./outputs/{LLM}/{MLLM}/{Method}/{Method}_{Dataset}.jsonl`
# Evaluation
## Recall rate
To evaluate the recall rate, you can run the following command:
```bash
python eval.py \
      --mode recall \
      --method MCCoT IICoT \
      --dataset_name PATH-VQA VQA-RAD Slake \
      --v_model LLava\
      --l_model ChatGLM Qwen2 Deepseek
```
This command evaluates the recall rates of ChatGLM and Qwen2 as LLMs, and LLava as an MLLM, using MCCoT and IICoT across all three datasets. 

Parameters can be adjusted to assess different combinations.
## Accuracy score
To evaluate the accuracy score, you can run the following command:
```bash
python eval.py \
      --mode acc \
      --method MMCoT IICoT \
      --dataset_name PATH-VQA VQA-RAD Slake \
      --v_model LLava QwenVL \
      --l_model GPT \
      --parallel \
      --max_workers 8
```
And to calculate and show the scaled score, please use:
```bash
python eval_show.py \
      --v_model LLava \
      --l_model GPT \
      --method DDCoT IICoT MMCoT \
      --dataset_name PATH-VQA VQA-RAD Slake
```