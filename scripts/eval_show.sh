echo 'llava'
python eval_show.py \
      --v_model LLava QwenVL \
      --l_model GPT Qwen2 \
      --method DDCoT IICoT MCCoT \
      --dataset_name PATH-VQA VQA-RAD Slake