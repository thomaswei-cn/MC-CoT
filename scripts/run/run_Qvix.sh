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
      --language_model_name GPT \
      --visual_model_name LLava \
      --dataset_name Slake \
      --slake_path /your_path_to_slake_dir

python run.py --method Qvix \
      --language_model_name GPT \
      --visual_model_name LLava \
      --dataset_name PATH-VQA \
      --path_vqa_path /your_path_to_path-vqa_dir

python run.py --method Qvix \
      --language_model_name GPT \
      --visual_model_name LLava \
      --dataset_name VQA-RAD \
      --vqa_rad_path /your_path_to_vqa-rad_dir