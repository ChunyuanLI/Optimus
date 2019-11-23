export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export GPU_ID=0,1


export GLUE_DIR=/workspace/data/datasets/glue_data/glue_data

python ./examples/run_glue_data_integration.py \
    --output_dir ../output/local_glue_data \
    --data_dir $GLUE_DIR\
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --percentage_per_label .5 \
    --use_freeze \
    --overwrite_output_dir
    
    