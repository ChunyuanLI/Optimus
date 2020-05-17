

## Pre-trained Models
We provide a series of pre-trained *Optimus* models of for different purpose, due to a trade-off between reconstruction capacity and prior regularization.

```bash
wget https://textae.blob.core.windows.net/optimus/$MODEL_DIR/$MODEL_NAME.zip
unzip $MODEL_NAME.zip -d $MODEL_NAME
```
`MODEL_DIR` and `MODEL_NAME` could be different values. We currently release the following models;

### A model with good latent space manipulation performance on SNLI dataset. 
```bash

wget https://textae.blob.core.windows.net/optimus/output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-full-31250.zip

mkdir output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted
mv checkpoint-full-31250.zip output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted
cd output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted

unzip checkpoint-full-31250.zip -d checkpoint-full-31250
```

### Play with the model

Interpolation

```bash
# interpolation

export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export TRAIN_FILE=../data/datasets/debug_data/train.txt
export TEST_FILE=../data/datasets/debug_data/test.txt
export GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
    --dataset Debug \
    --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted \
    --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted \
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
    --play_mode interpolation \
    --num_interpolation_steps 10

```


Reconstruction
 
```bash
# reconstrction
CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
    --dataset Debug \
    --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b0.5_d5_r00.5_ra0.25_length_weighted \
    --output_dir=../output/LM/Snli/768/philly_vae_snli_b0.5_d5_r00.5_ra0.25_length_weighted \
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
    --play_mode reconstrction
```


