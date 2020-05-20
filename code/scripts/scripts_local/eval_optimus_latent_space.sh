export PYTHONPATH="${PYTHONPATH}:/workspace/code"

# export TRAIN_FILE=../data/datasets/penn/train.txt
# export TEST_FILE=../data/datasets/penn/test.txt


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


export TRAIN_FILE=../data/datasets/debug_data/train.txt
export TEST_FILE=../data/datasets/debug_data/test.txt
export GPU_ID=1


# # interpolation from pre-trained model on wiki
# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
#     --dataset Debug \
#     --checkpoint_dir=../output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta1.0_d1.0_ro0.5_ra0.25_768_v2 \
#     --output_dir=../output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta1.0_d1.0_ro0.5_ra0.25_768_v2 \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1 \
#     --gloabl_step_eval 508523 \
#     --block_size 100 \
#     --max_seq_length 100 \
#     --latent_size 768 \
#     --play_mode interpolation \
#     --num_interpolation_steps 10


# # reconstruction from pre-trained model on wiki
# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
#     --dataset Debug \
#     --checkpoint_dir=../output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta0.0_d1.0_ro0.5_ra0.25_32_v2 \
#     --output_dir=../output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta0.0_d1.0_ro0.5_ra0.25_32_v2 \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1 \
#     --gloabl_step_eval 400000 \
#     --block_size 100 \
#     --max_seq_length 100 \
#     --latent_size 32 \
#     --play_mode reconstrction



# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
#     --dataset Debug \
#     --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
#     --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1 \
#     --gloabl_step_eval 31250 \
#     --block_size 100 \
#     --max_seq_length 100 \
#     --latent_size 768 \
#     --play_mode interpolation \
#     --num_interpolation_steps 10

# reconstrction
# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
#     --dataset Debug \
#     --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
#     --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1 \
#     --gloabl_step_eval 31250 \
#     --block_size 100 \
#     --max_seq_length 100 \
#     --latent_size 768 \
#     --play_mode reconstrction


# interact_with_user_input
CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
    --dataset Debug \
    --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
    --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --per_gpu_eval_batch_size=1 \
    --gloabl_step_eval 31250 \
    --block_size 100 \
    --max_seq_length 100 \
    --latent_size 768 \
    --interact_with_user_input \
    --play_mode analogy \
    --sent_source="a yellow cat likes to chase a long string ." \
    --sent_target="a yellow cat likes to chase a short string ." \
    --sent_input="a brown dog likes to eat long pasta ." \
    --degree_to_target=1.0
        


# interact_with_user_input
# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
#     --dataset Debug \
#     --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
#     --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --per_gpu_eval_batch_size=1 \
#     --gloabl_step_eval 31250 \
#     --block_size 100 \
#     --max_seq_length 100 \
#     --latent_size 768 \
#     --interact_with_user_input \
#     --play_mode interpolation \
#     --sent_source="a yellow cat likes to chase a short string ." \
#     --sent_target="a brown dog likes to eat his food very slowly ." \
#     --num_interpolation_steps=10


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