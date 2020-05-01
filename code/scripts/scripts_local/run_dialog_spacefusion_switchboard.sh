export PYTHONPATH="${PYTHONPATH}:/workspace/code"

export TRAIN_FILE=../data/datasets/switchboard/train.txt
export TEST_FILE=../data/datasets/switchboard/test.txt.1ref
export GENERATED_TEXT_FILE=../output/dialog/local-dialog-switchboard/eval_text_generation_results.txt

export GPU_ID=0,1

# CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_spacefusion_pretraining.py \
#     --dataset dailydialog \
#     --output_dir=../output/dialog/local-dialog-switchboard \
#     --encoder_model_type=bert \
#     --encoder_model_name_or_path=bert-base-cased \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=gpt2 \
#     --train_data_file=$TRAIN_FILE \
#     --do_generation \
#     --do_train \
#     --do_eval \
#     --beta 2.0 \
#     --ratio_zero .5 \
#     --ratio_increase 0.25 \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs 5.0 \
#     --save_steps 2000 \
#     --logging_steps 100 \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size 4 \
#     --block_size 512 \
#     --freeze_bert11 \
#     --per_gpu_eval_batch_size 1 \
#     --total_sents -1 \
#     --sents_per_cxt 10 \
#     --eval_generated_text_file $GENERATED_TEXT_FILE\
#     --checkpoint_dir ../output/philly_rr3scl_g8_vae_wikipedia_pretraining_beta_schedule_beta1.0_d1.0_ro0.5_ra0.25 \
#     --gloabl_step_eval 760000  \
#     --use_pretrained_model \
#     --use_pretrained_vae



export GENERATED_TEXT_PATH=philly-switchboard-epoch-5.0-beta-1.0
export GENERATED_TEXT_FILE=../output/dialog/$GENERATED_TEXT_PATH/eval_text_generation_results.txt

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_spacefusion_pretraining.py \
    --dataset switchboard \
    --output_dir=../output/dialog/$GENERATED_TEXT_PATH \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --beta 2.0 \
    --ratio_zero .5 \
    --ratio_increase 0.25 \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 1.0 \
    --save_steps 2000 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 4 \
    --block_size 512 \
    --freeze_bert11 \
    --per_gpu_eval_batch_size 1 \
    --total_sents -1 \
    --sents_per_cxt 10 \
    --eval_generated_text_file $GENERATED_TEXT_FILE\
    --checkpoint_dir ../output/dialog/$GENERATED_TEXT_PATH \
    --gloabl_step_eval 94000  \
    --use_pretrained_model

