echo "Deepseek"
export OPENAI_API_KEY=
export OPENAI_API_BASE=
conda activate M4

nohup python eval.py \
      --model deepseek-chat \
      --mode recall \
      --method qvix ddcot IICoT  M3 \
      --dataset_name PATH RAD Slake \
      --v_model llava\
      --l_model chatGLM qwen2 deepseek \
      --parallel \
      --max_workers 8


nohup python eval.py \
      --model gpt-3.5-turbo \
      --mode recall \
      --method M3-noana M3-nopth M3-norad M3 \
      --dataset_name PATH RAD Slake \
      --v_model llava \
      --parallel \
      --max_workers 8

python eval.py \
      --mode recall \
      --method M3 only ddcot cot ga \
      --dataset_name PATH RAD Slake \
      --v_model llava qwen qwen-max deepseek


python eval.py \
      --mode recall \
      --method cantor-med ccot check dga idealgpt M3-noana M3-nodes M3-noguide \
      M3-nopth M3-norad M3-v mmcot ps qvix \
      --dataset_name PATH RAD Slake \
      --v_model llava
