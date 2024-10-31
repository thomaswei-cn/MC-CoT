echo "Deepseek"
export Deepseek_API_KEY=sk-4e9afcccc35e4cc0bb2e6b55cc64f8b2
export Deepseek_API_BASE=https://api.deepseek.com

python eval.py \
      --mode recall \
      --method MCCoT IICoT \
      --dataset_name PATH RAD Slake \
      --v_model llava\
      --l_model gpt4 \

python eval.py \
      --mode acc \
      --method MCCoT IICoT \
      --dataset_name PATH RAD Slake \
      --v_model llava qwen-max \
      --l_model gpt3.5 \
      --parallel \
      --max_workers 8
