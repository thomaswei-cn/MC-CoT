conda activate M4

echo 'llava'
python eval_show.py \
      --v_model llava\
      --l_model deepseek \
      --method qvix ddcot IICoT M3 \
      --dataset_name PATH RAD Slake

python eval_show.py \
      --v_model llava \
      --method M3-noana M3-nopth M3-norad M3 \
      --dataset_name PATH RAD Slake \


echo 'qwen'
python eval_show.py \
      --v_model qwen \
      --method only cot ddcot ga M3 \
      --dataset_name PATH RAD Slake

echo 'qwen-max'
python eval_show.py \
      --v_model qwen-max \
      --method only cot ddcot ga M3 \
      --dataset_name PATH RAD Slake

echo 'deepseek'
python eval_show.py \
      --v_model deepseek \
      --method only cot ddcot ga M3 \
      --dataset_name PATH RAD Slake
