export PYTHONPATH="${PYTHONPATH}:/workspace/code"

# export TRAIN_FILE=../data/datasets/wikitext-2/train.txt
# export TEST_FILE=../data/datasets/wikitext-2/valid.txt
# export GPU_ID=0,1

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_encoding_generation.py \
#     --checkpoint_dir=../output/philly_clm_wiki2_0.0 \
#     --output_dir=../output/philly_clm_wiki2_0.0 \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1



# export TRAIN_FILE=../data/datasets/debug_data/train.txt
# export TEST_FILE=../data/datasets/debug_data/test.txt
# export GPU_ID=0,1

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_encoding_generation.py \
#     --checkpoint_dir=../output/local_lm_vae_debug_bert_gpt \
#     --output_dir=../output/local_lm_vae_debug_bert_gpt \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1 \
#     --gloabl_step_eval 400


export TRAIN_FILE=../data/datasets/snli_data/train.txt
export TEST_FILE=../data/datasets/snli_data/test.txt
export GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_encoding_generation.py \
    --dataset Snli \
    --checkpoint_dir=../output/philly_vae_snli_epoch20_b1.0_d0.5_r00.5_ra0.25 \
    --output_dir=../output/local_lm_vae_snli_bert_gpt \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --per_gpu_eval_batch_size=1 \
    --gloabl_step_eval 50000 \
    --block_size 100 \
    --max_seq_length 100 \
    --play_mode interpolation \
    --num_interpolation_steps 20





# export TRAIN_FILE=../data/datasets/debug_data/train.txt
# export TEST_FILE=../data/datasets/debug_data/test.txt
# export GPU_ID=1

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_encoding_generation.py \
#     --dataset Debug \
#     --checkpoint_dir=../output/local_lm_vae_debug_bert_gpt \
#     --output_dir=../output/local_lm_vae_debug_bert_gpt \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1 \
#     --gloabl_step_eval 800 \
#     --total_sents 10    