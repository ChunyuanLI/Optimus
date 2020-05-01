export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export GPU_ID=0,1

# export TRAIN_FILE=../data/datasets/debug_data/train.txt
# export TEST_FILE=../data/datasets/debug_data/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --output_dir=../output/LM/local_lm_vae_debug_optimus \
#     --dataset Debug \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --beta 1.0 \
#     --ratio_zero 0.5 \
#     --ratio_increase 0.25 \
#     --do_train \
#     --do_eval \
#     --fb_mode 1 \
#     --dim_target_kl 0.5\
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 100.0 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=10 \
#     --block_size 100 \
#     --use_pretrained_model \
#     --use_pretrained_vae \
#     --checkpoint_dir ../output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta1.0_d1.0_ro0.5_ra0.25_32_v2 \
#     --gloabl_step_eval 320000



# export TRAIN_FILE=../data/datasets/yelp_data/train.txt
# export TEST_FILE=../data/datasets/yelp_data/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --output_dir=../output/local_lm_vae_yelp_bert_gpt \
#     --dataset Yelp \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
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
#     --per_gpu_train_batch_size=2 \
#     --block_size 300

# export TRAIN_FILE=../data/datasets/yahoo_data/train.txt
# export TEST_FILE=../data/datasets/yahoo_data/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --output_dir=../output/local_lm_vae_yahoo_bert_gpt \
#     --dataset Yahoo \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
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
#     --per_gpu_train_batch_size=2 \
#     --block_size 300

# export TRAIN_FILE=../data/datasets/snli_data/train.txt
# export TEST_FILE=../data/datasets/snli_data/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --output_dir=../output/local_lm_vae_snli_bert_gpt \
#     --dataset Snli \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
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
#     --per_gpu_train_batch_size=10 \
#     --block_size 100

export TRAIN_FILE=../data/datasets/penn/train.txt
export TEST_FILE=../data/datasets/penn/test.txt

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
    --output_dir=../output/local_lm_vae_penn_bert_gpt \
    --dataset Penn \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --beta 1.0 \
    --ratio_zero 0.5 \
    --ratio_increase 0.25 \
    --do_train \
    --do_eval \
    --fb_mode 1 \
    --dim_target_kl 0.5\
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 1.0 \
    --save_steps 1000 \
    --logging_steps 1000 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=10 \
    --block_size 100 \
    --use_pretrained_model \
    --use_pretrained_vae \
    --checkpoint_dir ../output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta1.0_d1.0_ro0.5_ra0.25_32_v2 \
    --gloabl_step_eval 320000

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --output_dir=../output/local_lm_vae_penn_bert_gpt \
#     --dataset Penn \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --beta 1.0 \
#     --ratio_zero 0.5 \
#     --ratio_increase 0.25 \
#     --do_train \
#     --do_eval \
#     --fb_mode 1 \
#     --dim_target_kl 0.5\
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=10 \
#     --block_size 100

# export TRAIN_FILE=../data/datasets/wikipedia/wikipedia.segmented.nltk.txt
# export TEST_FILE=../data/datasets/wikipedia/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_pretraining.py \
#     --output_dir=../output/local_lm_vae_wikipedia_bert_gpt \
#     --dataset wikipedia \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-uncased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --beta 1.0 \
#     --ratio_zero 0.5 \
#     --ratio_increase 0.25 \
#     --do_train \
#     --fb_mode 1 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=20 \
#     --block_size 100



# export TRAIN_FILE=../data/datasets/news_data/train.txt
# export TEST_FILE=../data/datasets/news_data/test.txt


# export TRAIN_FILE=../data/datasets/glue_data/glue_data/YELP/train.txt
# export TEST_FILE=../data/datasets/glue_data/glue_data/YELP/test.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --dataset News \
#     --checkpoint_dir ../output/philly_scl_b16_g8_vae_wikipedia_pretraining_b0.0_d1.0_r01.0_ra0.1_200 \
#     --gloabl_step_eval 880000 \
#     --output_dir=../output/local_lm_vae_news_bert_gpt \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --do_train \
#     --do_eval \
#     --beta 1.0 \
#     --dim_target_kl 1.0 \
#     --ratio_zero .0 \
#     --ratio_increase 0.25 \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 5000 \
#     --logging_steps 200 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size 16 \
#     --block_size 256 \
#     --use_pretrained_model

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --dataset News \
#     --output_dir=../output/local_lm_vae_bert_gpt_init_768 \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --do_train \
#     --do_eval \
#     --beta 1.0 \
#     --dim_target_kl 1.0 \
#     --ratio_zero .0 \
#     --ratio_increase 0.25 \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 5000 \
#     --logging_steps 200 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size 16 \
#     --block_size 256 \
#     --latent_size 768 \
#     --save_bert_gpt_init


# export TRAIN_FILE=../data/datasets/glue_data/train.txt
# export TEST_FILE=../data/datasets/glue_data/train.txt

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
#     --dataset Glue \
#     --checkpoint_dir ../output/philly_scl_b16_g8_vae_wikipedia_pretraining_b0.0_d1.0_r01.0_ra0.1_200 \
#     --gloabl_step_eval 880000 \
#     --output_dir=../output/local_lm_vae_glue_bert_gpt \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --do_train \
#     --beta 1.0 \
#     --dim_target_kl 1.0 \
#     --ratio_zero .5 \
#     --ratio_increase 0.25 \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 1.0 \
#     --save_steps 5000 \
#     --logging_steps 200 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size 16 \
#     --block_size 256 \
#     --use_pretrained_model
