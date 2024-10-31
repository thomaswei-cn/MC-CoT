echo "GPT"
export OPENAI_API_KEY=
export OPENAI_API_BASE=

echo "Deepseek"
export Deepseek_API_KEY=
export Deepseek_API_BASE=

echo "Qwen2"
export Dashscope_API_KEY=
export Qwen2_API_BASE=

python run.py --method Method.DDCoT \
      --language_model_name Engine.GPT \
      --visual_model_name Engine.LLava \
      --dataset_name Slake \
      --slake_path ../data/Slake1.0 \
      --output_file_path ./outputs/qwen2/llava/ddcot/ddcot_Slake.jsonl \
      --shuffle \
      --truncation_50 \
      --max_retries 1


python run.py --method Method.DDCoT \
      --language_model_name Engine.GPT \
      --visual_model_name Engine.LLava \
      --dataset_name PATH-VQA \
      --path_vqa_path ../data/PATH-VQA \
      --output_file_path ./outputs/gpt4/llava/ddcot/ddcot_PATH-VQA.jsonl \
      --shuffle \
      --truncation_50 \
      --max_retries 1

python run.py --method Method.DDCoT \
      --language_model_name Engine.GPT \
      --visual_model_name Engine.LLava \
      --dataset_name VQA-RAD \
      --vqa_rad_path ../data/VQA-RAD \
      --output_file_path ./outputs/gpt4/llava/ddcot/ddcot_VQA-RAD.jsonl \
      --shuffle \
      --truncation_50 \
      --max_retries 1