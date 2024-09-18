echo "GPT"
export OPENAI_API_KEY=
export OPENAI_API_BASE=

echo "Deepseek"
export Deepseek_API_KEY=
export Deepseek_API_BASE=

echo "Qwen2"
export Dashscope_API_KEY=
export Qwen2_API_BASE=

nohup python run.py --method Method.MMCoT \
      --language_model_name Engine.GPT \
      --visual_model_name Engine.LLava \
      --dataset_name Slake \
      --slake_path /data/M4_data/Slake1.0 \
      --question_type open \
      --output_file_path ./outputs/llava/mmcot/mmcot_Slake_open.jsonl > MMCoT_Slake_open.txt


nohup python run.py --method Method.MMCoT \
      --language_model_name Engine.GPT \
      --visual_model_name Engine.LLava \
      --dataset_name PATH-VQA \
      --path_vqa_path /data/M4_data/PATH-VQA \
      --question_type open \
      --output_file_path ./outputs/MMCoT/open/MMCoT_PATH_open.jsonl > MMCoT_PATH_open.txt

nohup python run.py --method Method.MMCoT \
      --language_model_name Engine.GPT \
      --visual_model_name Engine.LLava \
      --dataset_name VQA-RAD \
      --vqa_rad_path /data/M4_data/VQA-RAD \
      --question_type open \
      --output_file_path ./outputs/MMCoT/open/MMCoT_RAD_open.jsonl > MMCoT_RAD_open.txt