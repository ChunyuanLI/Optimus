export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export GPU_ID=0,1


export GLUE_DIR=/workspace/data/datasets/glue_data/glue_data
export TASK_NAME=YELP # SST-2 # CoLA  # SST-2 # MRPC

# python ./examples/run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-cased \
#     --task_name $TASK_NAME \
#     --do_eval \
#     --do_lower_case \
#     --save_steps 200 \
#     --data_dir $GLUE_DIR/$TASK_NAME \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=32   \
#     --per_gpu_train_batch_size=32   \
#     --learning_rate 2e-5 \
#     --num_train_epochs 50.0 \
#     --percentage_per_label .5 \
#     --sample_per_label 10000 \
#     --output_dir ../output/local_features_$TASK_NAME/ \
#     --use_freeze \
#     --overwrite_output_dir \
#     --eval_all_checkpoints \
#     --collect_feature
    
python ./examples/run_glue_vae.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --save_steps 200 \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --num_train_epochs 50.0 \
    --percentage_per_label .5 \
    --sample_per_label 10000 \
    --output_dir ../output/local_features_$TASK_NAME/ \
    --use_freeze \
    --overwrite_output_dir \
    --collect_feature \
    --checkpoint_dir ../output/philly_rr1_vae_yelp_short_epoch1.0_b1.0_d1.0_r00.5_ra0.25 \
    --gloabl_step_eval 6939 
        