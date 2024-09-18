echo "GPT"
export OPENAI_API_KEY=
export OPENAI_API_BASE=

echo "Deepseek"
export Deepseek_API_KEY=
export Deepseek_API_BASE=

echo "Qwen2"
export Dashscope_API_KEY=
export Qwen2_API_BASE=

nohup python run.py --method Method.DDCoT \
      --language_model_name Engine.Qwen2 \
      --visual_model_name Engine.LLava \
      --dataset_name Slake \
      --slake_path /data/M4_data/Slake1.0 \
      --output_file_path ./outputs/qwen2/llava/ddcot/ddcot_Slake_open.jsonl > ddcot_Slake_open.txt


nohup python run.py --method Method.DDCoT \
      --language_model_name Engine.Qwen2 \
      --visual_model_name Engine.LLava \
      --dataset_name PATH-VQA \
      --path_vqa_path /data/M4_data/PATH-VQA \
      --output_file_path ./outputs/qwen2/llava/ddcot/ddcot_PATH_open.jsonl > ddcot_PATH_open.txt

nohup python run.py --method Method.DDCoT \
      --language_model_name Engine.Qwen2 \
      --visual_model_name Engine.LLava \
      --dataset_name VQA-RAD \
      --vqa_rad_path /data/M4_data/VQA-RAD \
      --output_file_path ./outputs/qwen2/llava/ddcot/ddcot_RAD_open.jsonl > ddcot_RAD_open.txt