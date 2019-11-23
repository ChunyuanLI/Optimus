export PYTHONPATH="${PYTHONPATH}:/workspace/code"

export TRAIN_FILE=../data/datasets/debug_data/train.txt
export TEST_FILE=../data/datasets/debug_data/test.txt
export GPU_ID=0,1

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
    --dataset Debug \
    --output_dir=../output/local_lm_vae_debug_bert_gpt \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --do_train \
    --do_eval \
    --beta 1.0 \
    --ratio_zero .5 \
    --ratio_increase 0.25 \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 5.0 \
    --save_steps 20 \
    --logging_steps 4 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 3 \
    --block_size 50
