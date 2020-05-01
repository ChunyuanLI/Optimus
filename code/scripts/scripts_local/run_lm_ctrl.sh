export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export GPU_ID=0,1
# export TRAIN_FILE=../data/datasets/wikitext-2/train.txt
# export TEST_FILE=../data/datasets/wikitext-2/valid.txt



export TRAIN_FILE=../data/datasets/yelp_style/sentiment.train.text
export TEST_FILE=../data/datasets/yelp_style/sentiment.test.text.1000sents



# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_label_ctrl_gen.py \
#     --output_dir ../output/local_lm_vae_label_ctrl_gen \
#     --checkpoint_dir ../output/philly_cara_yelp_50.0 \
#     --gloabl_step_eval 43650  \
#     --dataset Yelp \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --num_train_epochs 1.0 \
#     --overwrite_output_dir 1 \
#     --per_gpu_train_batch_size=32 \
#     --block_size 300 \
#     --do_eval




CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_label_ctrl_gen.py \
    --output_dir ../output/local_lm_vae_label_ctrl_gen \
    --checkpoint_dir ../output/local_lm_vae_label_ctrl_gen \
    --gloabl_step_eval 6989  \
    --use_pretrained_model \
    --dataset Yelp \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --save_steps 1000 \
    --logging_steps 1000 \
    --num_train_epochs 1.0 \
    --overwrite_output_dir 1 \
    --per_gpu_train_batch_size=32 \
    --block_size 300 \
    --do_eval



# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_label_ctrl_gen.py \
#     --output_dir ../output/local_lm_vae_label_ctrl_gen \
#     --checkpoint_dir ../output/philly_rr3scl_g8_vae_wikipedia_pretraining_beta_schedule_beta1.0_d1.0_ro0.5_ra0.25 \
#     --gloabl_step_eval 760000  \
#     --use_pretrained_model \
#     --use_pretrained_vae \
#     --dataset Yelp \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --num_train_epochs 1.0 \
#     --overwrite_output_dir 1 \
#     --per_gpu_train_batch_size=32 \
#     --block_size 300 \
#     --do_eval \
#     --do_train 


# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_label_ctrl_gen.py \
#     --output_dir ../output/local_lm_vae_label_ctrl_gen \
#     --checkpoint_dir ../output/philly_scl_b16_g8_vae_wikipedia_pretraining_b0.0_d1.0_r01.0_ra0.1_200 \
#     --dataset Yelp \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --gloabl_step_eval 880000 \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --save_steps 1000 \
#     --logging_steps 1000 \
#     --num_train_epochs 1.0 \
#     --overwrite_output_dir 1 \
#     --per_gpu_train_batch_size=32 \
#     --use_pretrained_model  \
#     --block_size 300 \
#     --do_train \
#     --do_eval 



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
