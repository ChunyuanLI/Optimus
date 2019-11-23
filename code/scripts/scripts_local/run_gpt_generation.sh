
export GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2