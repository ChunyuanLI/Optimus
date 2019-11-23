
export TRAIN_FILE=../data/datasets/wikitext-2/train.txt
export TEST_FILE=../data/datasets/wikitext-2/valid.txt
export GPU_ID=0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_lm_causal_pretraining.py \
    --output_dir=../output/local_lm_causal_wiki2_bert_gpt \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-uncased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 1.0 \
    --save_steps 200 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=1