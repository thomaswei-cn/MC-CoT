echo "Deepseek"
export Deepseek_API_KEY=
export Deepseek_API_BASE=

python eval.py \
      --mode recall \
      --method MCCoT IICoT \
      --dataset_name PATH RAD Slake \
      --v_model llava\
      --l_model chatGLM qwen2 deepseek \

python eval.py \
      --mode acc \
      --method MCCoT IICoT \
      --dataset_name PATH RAD Slake \
      --v_model llava qwen-max \
      --l_model gpt3.5 \
      --parallel \
      --max_workers 8
