export PYTHONPATH="${PYTHONPATH}:/workspace/code"

export TRAIN_FILE=../data/datasets/penn/train.txt
export TEST_FILE=../data/datasets/penn/test.txt
export GPU_ID=0,1

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
    --dataset Debug \
    --checkpoint_dir=../output/inject_latent/philly_vae_penn_b0.0_emb_1_mem_0 \
    --output_dir=../output/inject_latent/philly_vae_penn_b0.0_emb_1_mem_0 \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --do_eval_rec \
    --beta 0.0 \
    --ratio_zero .5 \
    --ratio_increase 0.25 \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 2.0 \
    --save_steps 20 \
    --logging_steps 4 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 10 \
    --gloabl_step_eval 4 \
    --block_size 100 \
    --latent_size 32 \
    --latent_as_gpt_emb 1 \
    --latent_as_gpt_memory 0 \
    --gloabl_step_eval 13145



# export TRAIN_FILE=../data/datasets/debug_data/train.txt
# export TEST_FILE=../data/datasets/debug_data/test.txt
# export GPU_ID=0,1

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --dataset Debug \
#     --checkpoint_dir=../output/philly_rr1_vae_wikipedia_pretraining_b0.0_d1.0_r01.0_ra0.1 \
#     --output_dir=../output/local_lm_vae_debug_bert_gpt \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --do_train \
#     --do_eval \
#     --beta 1.0 \
#     --ratio_zero .5 \
#     --ratio_increase 0.25 \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 2.0 \
#     --save_steps 20 \
#     --logging_steps 4 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size 1 \
#     --gloabl_step_eval 4 \
#     --block_size 50 \
#     --latent_size 32 \
#     --latent_as_gpt_memory 0

