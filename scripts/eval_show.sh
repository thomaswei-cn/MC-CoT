echo 'llava'
python eval_show.py \
      --v_model llava qwen \
      --l_model gpt3.5 qwen2 \
      --method DDCoT IICoT MCCoT \
      --dataset_name PATH RAD Slake