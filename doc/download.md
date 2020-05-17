

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


Here are some results you can see from the model:

Interpolation
```
0 
a woman is riding a moped on a street with large trees and other riders ride over it as some sort of bob lights are passing behind her.
1 
 a woman is riding a moped on a street with several buildings that run in front of and polluting the behind her ears.
2 
 a woman is riding a small pig in front of a street with billows and speakers that lead to the windowspan on them.
3 
 a woman on a street riding mule is holding two wheels in front of a large white tulips while it sews the air behind them.
4 
 a man in a ponytail is riding two of the houses visible on the street while hanging traffic cones over them.
5 
 one man riding a van in front of the street is shining blue curtains, while another man holding on to the moped wires.
6 
 one man in a blue shirt and black riding a windowless cart are riding over the others next to them hanging on the river.
7 
 two men in yellow riding a wave just standing on the side of the building keeping track of another man.
8 
 two men in matching blue shirt standing on the roof of a car one it reading a trolley car.
9 
 two men in the blue one car standing on the window dreaming of hanging another car passing by.
10 
 two men in blue holding each other standing the window of a one wheel bicycle trying out the tube.

```


Reconstruction (beta = 0.5)
```
a football coach putting his arm on one of his player's shoulder.  
 a football player putting his arm on another's shoulder, this one hispanic.

 a girl wearing a white shirt and blue jeans jumping off a rock into the sand.  
 a girl wearing a blue shirt and white jeans jumping off a rock into the sand.

 a group of cheerleaders are making a human pyramid at a basketball game.  
 a group of cheerleaders are making a human pyramid at a basketball game.

 a man is looking over things that are in a local market.  
 a man is looking over things that are in a local market.

 a woman is riding a moped on a street with many other people riding mopeds behind her, while streamers and banners hang in the trees overhead.  
 a woman is riding a moped on a street with many other people behind her, as well as small banners and horns riding in the background.

 group of people in the wilderness packing boxes full of food.  
 group of people in the wilderness packing boxes full of food.

 man tries to impress girl by diving into the water.  
 man tries to impress girl by diving into the water.

 people in the middle of city street surrounded by large buildings.  
 people in the middle of city streets surrounded by large buildings.

 there are girls that are on the ice practicing their ice dancing.  
 there are girls that are on the ice practicing their ice dancing.

 there are two young boys with shovels who are outside and are digging in the dirt.  
 there are two young boys with shovels who are outside and are digging in the dirt.

 two men in blue one standing the other hanging onto the window of a tram car just looking out.  
 two men in all blue standing the windowless car next to another man riding a blue roller coaster.
```



Reconstruction (beta = 1.0)
```
a football coach putting his arm on one of his player's shoulder.  
 a football player extending his hand on the team football.

 a girl wearing a white shirt and blue jeans jumping off a rock into the sand.  
 a girl wearing a blue shirt and blue jeans jumping off the rock into the ocean.

 a group of cheerleaders are making a human pyramid at a basketball game.  
 a group of girls are creating a purple basketball <unk> at a giant auditorium.

 a man is looking over things that are in a local market.  
 a man is looking over things in a local marketplace, looking very busy.

 a woman is riding a moped on a street with many other people riding mopeds behind her, while streamers and banners hang in the trees overhead.  
 a woman is riding a bike on a grassy lot with other people, riding flag poles and asian flags in front of it.

 group of people in the wilderness packing boxes full of food.  
 group of people in the packing room packing stuff.

 man tries to impress girl by diving into the water.  
 man tries to impress the girl by swimming underwater.

 people in the middle of city street surrounded by large buildings.  
 people in high, urban city blocks surrounding the building.

 there are girls that are on the ice practicing their ice dancing.  
 there are two girls who are practicing the ice skating in the snow.

 there are two young boys with shovels who are outside and are digging in the dirt.  
 there are two young children with shovels and shovels in the dirt.

 two men in blue one standing the other hanging onto the window of a tram car just looking out.  
 two men in yellow overalls standing next to the wheel of a blue car while they look on.

```
