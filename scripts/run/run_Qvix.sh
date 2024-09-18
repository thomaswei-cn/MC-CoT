echo "GPT"
export OPENAI_API_KEY=
export OPENAI_API_BASE=

echo "Deepseek"
export Deepseek_API_KEY=
export Deepseek_API_BASE=

echo "Qwen2"
export Dashscope_API_KEY=
export Qwen2_API_BASE=

python run.py --method Qvix \
      --language_model_name Qwen2 \
      --visual_model_name LLava \
      --dataset_name Slake \
      --slake_path /data/M4_data/Slake1.0

python run.py --method Qvix \
      --language_model_name Qwen2 \
      --visual_model_name LLava \
      --dataset_name PATH-VQA \
      --path_vqa_path /data/M4_data/PATH-VQA

python run.py --method Qvix \
      --language_model_name Qwen2 \
      --visual_model_name LLava \
      --dataset_name VQA-RAD \
      --vqa_rad_path /data/M4_data/VQA-RAD