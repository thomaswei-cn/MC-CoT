echo "Deepseek"
export Deepseek_API_KEY=
export Deepseek_API_BASE=

python eval.py \
      --mode recall \
      --method MCCoT IICoT \
      --dataset_name PATH-VQA VQA-RAD Slake \
      --v_model LLava\
      --l_model ChatGLM Qwen2 Deepseek \

python eval.py \
      --mode acc \
      --method MCCoT IICoT \
      --dataset_name PATH-VQA VQA-RAD Slake \
      --v_model LLava QwenVL-Max \
      --l_model GPT \
      --parallel \
      --max_workers 8
