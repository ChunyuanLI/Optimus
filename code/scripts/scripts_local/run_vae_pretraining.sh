export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export GPU_ID=0,1

export TRAIN_FILE=../data/datasets/wikipedia_json_64/

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_pretraining.py \
#     --output_dir=../output/local_lm_vae_wikipedia_pretraining \
#     --dataset wikipedia \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --beta 0.0 \
#     --ratio_zero 1.0 \
#     --ratio_increase 0.1 \
#     --do_train \
#     --fb_mode 1 \
#     --train_data_file=$TRAIN_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 10000 \
#     --logging_steps 1000 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=8 \
#     --block_size 256

CUDA_VISIBLE_DEVICES=$GPU_ID python  -m torch.distributed.launch --nproc_per_node 2 examples/big_ae/run_lm_vae_pretraining_distributed.py \
    --output_dir=../output/local_lm_vae_wikipedia_pretraining \
    --dataset wikipedia \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --beta 0.0 \
    --ratio_zero 1.0 \
    --ratio_increase 0.1 \
    --do_train \
    --fb_mode 1 \
    --train_data_file=$TRAIN_FILE \
    --num_train_epochs 1.0 \
    --save_steps 10000 \
    --logging_steps 1000 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=8 \
    --block_size 256
