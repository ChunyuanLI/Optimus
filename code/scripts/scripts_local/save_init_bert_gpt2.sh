export PYTHONPATH="${PYTHONPATH}:/workspace/code"

export TRAIN_FILE=../data/datasets/penn/train.txt
export TEST_FILE=../data/datasets/penn/test.txt
export GPU_ID=0,1

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_vae_training.py \
    --dataset Penn \
    --checkpoint_dir=../output/local_bert_gpt_init \
    --output_dir=../output/local_bert_gpt_init \
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
    --num_train_epochs 2.0 \
    --save_steps 20 \
    --logging_steps 4 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 1 \
    --gloabl_step_eval 60000 \
    --block_size 128 \
    --latent_as_gpt_emb 1 \
    --latent_as_gpt_memory 1 \
    --save_bert_gpt_init \
    --latent_size 768
    # --use_pretrained_model
