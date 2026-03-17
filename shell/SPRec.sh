# echo "Starting SPRec pipeline for Cluster Exposure DPO..."
# CUDA_VISIBLE_DEVICES=1 python "scripts/run_sft.py"
gpu1=$1; 
base_config_path="../configs/SmolLM2-135M-Instruct_origin_setup/SPRec"
code_dir="re_run_args_scripts"
# base_config_path="../configs/Llama-3.2-1B-Instruct/SPRec"
echo "Finished SFT. Starting self-play generation..."
CUDA_VISIBLE_DEVICES=$gpu1 python "${code_dir}/run_generate_selfplay_args.py" --config_path "${base_config_path}/selfplay_config.yml"
echo "Finished self-play generation. Starting DPO training..."
CUDA_VISIBLE_DEVICES=$gpu1 python "${code_dir}/run_dpo_args.py" --config_path "${base_config_path}/dpo_config.yml"
echo "Finished DPO training. Starting prediction generation using ${base_config_path}/predict_config.yml..."
CUDA_VISIBLE_DEVICES=$gpu1 python "${code_dir}/run_generate_predictions_args.py" --config_path "${base_config_path}/predict_config.yml"
echo "Finished prediction generation. Starting evaluation..."
CUDA_VISIBLE_DEVICES=$gpu1 python "${code_dir}/run_evaluate_args.py" --config_path "${base_config_path}/eval_config.yml"



# gpu1=$1; 
# category="Goodreads"



# # for category in "MovieLens"  "Goodreads" "CDs_and_Vinyl" "Steam"
# for base_model in "SmolLM2-360M-Instruct"   
# do

#     echo ---------------------- SFT for category $category starting! ---------------------- 
#     train_dataset="./data/raw/${category}/train.json"
#     valid_dataset="./data/raw/${category}/valid.json"
#     output_dir="./experiments/model/sft_model_${base_model}_${category}"
#     base_model="HuggingFaceTB/${base_model}"

#     mkdir -p $output_dir

#     # Match gpu 4 to 1, gradient_accumulation_steps 16*4=64 effective batch size
#     CUDA_VISIBLE_DEVICES=$gpu1 python ./train/sft.py \
#         --output_dir $output_dir\
#         --base_model $base_model \
#         --train_dataset $train_dataset \
#         --valid_dataset $valid_dataset \
#         --train_sample_size $sample \
#         --wandb_project SFT_${category}_${sample} \
#         --wandb_name SFT_${category}_${sample} \
#         --gradient_accumulation_steps 16 \
#         --batch_size 4 \
#         --num_train_epochs 4 \
#         --learning_rate 0.0003 \
#         --cutoff_len 512 

#     bash ./shell/eval_single_file.sh  $gpu1 \
#                                             $base_model \
#                                             $output_dir \
#                                             $category \
#                                             $topk
# done
