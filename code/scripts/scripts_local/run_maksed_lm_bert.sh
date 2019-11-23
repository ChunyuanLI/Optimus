
export TRAIN_FILE=../data/datasets/wikitext-2/train.txt
export TEST_FILE=../data/datasets/wikitext-2/valid.txt
export GPU_ID=0,1

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/run_lm_finetuning.py \
#     --output_dir=../output/local_mlm_wiki2 \
#     --model_type=roberta \
#     --model_name_or_path=roberta-base \
#     --do_train \
#     --train_data_file=$TRAIN_FILE \
#     --do_eval \
#     --eval_data_file=$TEST_FILE \
#     --mlm \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=2


CUDA_VISIBLE_DEVICES=$GPU_ID python examples/run_lm_finetuning.py \
    --output_dir=../output/local_mlm_wiki2 \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=2