export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export GPU_ID=0,1


export TRAIN_FILE=../data/datasets/yahoo_data/train.txt
export TEST_FILE=../data/datasets/yahoo_data/test.txt

# export GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
    --output_dir=../output/philly_vae_yahoo_b0.25_d0.01_r00.5_ra0.25 \
    --dataset Yahoo \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 1.0 \
    --save_steps 200 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --evaluate_during_training \
    --per_gpu_train_batch_size=1 \
    --gloabl_step_eval 6250


# export TRAIN_FILE=../data/datasets/snli_data/train.txt
# export TEST_FILE=../data/datasets/snli_data/test.txt

# export GPU_ID=0

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_pretraining.py \
#     --output_dir=../output/local_lm_vae_snli_bert_gpt \
#     --dataset Snli \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --do_eval \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 200 \
#     --logging_steps 100 \
#     --overwrite_output_dir \
#     --evaluate_during_training \
#     --per_gpu_train_batch_size=1 \
#     --gloabl_step_eval 12000


# export TRAIN_FILE=../data/datasets/debug_data/train.txt
# export TEST_FILE=../data/datasets/debug_data/test.txt

# export GPU_ID=0

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_pretraining.py \
#     --output_dir=../output/local_lm_vae_debug_bert_gpt \
#     --dataset Snli \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --do_eval \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 200 \
#     --logging_steps 100 \
#     --overwrite_output_dir \
#     --evaluate_during_training \
#     --per_gpu_train_batch_size=1 \
#     --gloabl_step_eval 200
