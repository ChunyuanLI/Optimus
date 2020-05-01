export PYTHONPATH="${PYTHONPATH}:/workspace/code"

export INPUT_FILE_PATH=../data/datasets/wikipedia_json_64/
export OUTPUT_FILE_PATH=../data/datasets/wikipedia_json_64_filtered/
export OUTPUT_DIR=./output/data_preprocessing/log_wikipedia_overlength_filtering/
export GPU_ID=0,1

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_data_filtering.py \
    --dataset wikipedia \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --input_file_path=$INPUT_FILE_PATH \
    --output_file_path=$OUTPUT_FILE_PATH \
    --output_dir=$OUTPUT_DIR \
    --do_train \
    --do_eval \
    --beta 1.0 \
    --ratio_zero .5 \
    --ratio_increase 0.25 \
    --num_train_epochs 1.0 \
    --save_steps 20 \
    --logging_steps 4 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 2 \
    --gloabl_step_eval 4 \
    --block_size 50
