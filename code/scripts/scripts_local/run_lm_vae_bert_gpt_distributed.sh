export PYTHONPATH="${PYTHONPATH}:/workspace/code"

# export TRAIN_FILE=../data/datasets/wikitext-2/train.txt
# export TEST_FILE=../data/datasets/wikitext-2/valid.txt
export GPU_ID=0,1
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node 2 examples/big_ae/run_lm_vae_pretraining_phdist.py \
--num_train_epochs 1.0 --beta 0.0 --dim_target_kl 1.0 --ratio_zero 0.5 --ratio_increase 0.25 --latent_size 32 --dataset wikipedia \
--per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 1 --block_size 128 \
--output_dir ../output/pretrain/debug/g2_base_vae_wikipedia_pretraining_beta_schedule_beta0.0_d1.0_ro0.5_ra0.25_32 \
--encoder_model_type bert --encoder_model_name_or_path ../data/models/local_bert_gpt_init/initial-models-tokenization-enoder-32 \
--decoder_model_type gpt2 --decoder_model_name_or_path ../data/models/local_bert_gpt_init/initial-models-tokenization-decoder-32 \
--do_train --train_data_file ../data/datasets/wikipedia_json_64_filtered --overwrite_output_dir --save_steps 20000 --logging_steps 100 --use_beta_schedule


# export TRAIN_FILE=../data/datasets/yelp_data/train.txt
# export TEST_FILE=../data/datasets/yelp_data/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_pretraining.py \
#     --output_dir=../output/local_lm_vae_yelp_bert_gpt \
#     --dataset Yelp \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --do_train \
#     --do_eval \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=2


# export TRAIN_FILE=../data/datasets/yahoo_data/train.txt
# export TEST_FILE=../data/datasets/yahoo_data/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_pretraining.py \
#     --output_dir=../output/local_lm_vae_yahoo_bert_gpt \
#     --dataset Yahoo \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --beta 0.25 \
#     --ratio_zero 0.5 \
#     --ratio_increase 0.1 \
#     --do_train \
#     --do_eval \
#     --fb_mode 1 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=2


# export TRAIN_FILE=../data/datasets/snli_data/train.txt
# export TEST_FILE=../data/datasets/snli_data/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node 2 \
#     examples/big_ae/run_lm_vae_pretraining.py \
#     --output_dir=../output/local_lm_vae_snli_bert_gpt_distributed \
#     --dataset Snli \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --beta 1.0 \
#     --ratio_zero 0.5 \
#     --ratio_increase 0.25 \
#     --do_train \
#     --do_eval \
#     --fb_mode 1 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=30 \
#     --block_size 100
