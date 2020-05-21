

## Pre-trained Models for SNLI dataset
_Note: We provide a series of pre-trained *Optimus* models of for different purpose, due to a trade-off between reconstruction capacity and prior regularization._

```bash
wget https://textae.blob.core.windows.net/optimus/$MODEL_DIR/$MODEL_NAME.zip
unzip $MODEL_NAME.zip -d $MODEL_NAME
```
`MODEL_DIR` and `MODEL_NAME` could be different values. We currently release the following models;


Play with our [`demo`](http://40.71.23.172:8899/), including sentence interpolation and analogy.

## A model with good latent space manipulation performance on SNLI dataset. 

To download a model with with beta=1.0 at the following link:
https://textae.blob.core.windows.net/optimus/output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250.zip

Each zip file contains three folders: `full`, `encoder` and `decoder`.

```bash

wget https://textae.blob.core.windows.net/optimus/output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250.zip

mkdir -p output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted
mv checkpoint-31250.zip output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted
cd output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted

unzip checkpoint-31250.zip
```

Similarly, one may download models with with beta=0.0 and beta=0.5 at the following links:

beta=0.0 
https://textae.blob.core.windows.net/optimus/output/LM/Snli/768/philly_vae_snli_b0.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250.zip

beta=0.5 
https://textae.blob.core.windows.net/optimus/output/LM/Snli/768/philly_vae_snli_b0.5_d5_r00.5_ra0.25_length_weighted/checkpoint-31250.zip


### Play with user input sentences

The main training script is [`run_latent_generation.py`](../code/examples/big_ae/run_latent_generation.py) and conducts the fine-tuning loop, taking the following options (among others) as arguments:

- `--interact_with_user_input`: it specifies the program will take user inputs
- `--play_mode`: Two modes are supported: [`analogy`, `interpolation`]
- `--sent_source` and `--sent_target`: the source and target sentences to interpolate in between, or to make an analogy
- `--num_interpolation_steps`: the number of interpolated sentences between source and target sentences 
- `--sent_input`: the input sentence that will be re-written with the analogy specified by the source and target sentences
- `--degree_to_target`: (float type), the degree to which the analogy will made, default value is 1.0. 

Here are two examples:

```
export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export TRAIN_FILE=../data/datasets/debug_data/train.txt
export TEST_FILE=../data/datasets/debug_data/test.txt
export GPU_ID=1

# analogy
CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
    --dataset Debug \
    --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
    --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
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
    --interact_with_user_input \
    --play_mode analogy \
    --sent_source="a yellow cat likes to chase a long string ." \
    --sent_target="a yellow cat likes to chase a short string ." \
    --sent_input="a brown dog likes to eat long pasta ." \
    --degree_to_target=1.0
    
# interpolation    
CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
    --dataset Debug \
    --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
    --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
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
    --interact_with_user_input \
    --play_mode interpolation \
    --sent_source="a yellow cat likes to chase a short string ." \
    --sent_target="a brown dog likes to eat his food very slowly ." \
    --num_interpolation_steps=10

```
_Acknowledgement: the user interaction mode is updated with the suggestion from [summerstay](https://github.com/summerstay), in an issue [thread](https://github.com/ChunyuanLI/Optimus/issues/4)_

### Play with the my debugging dataset, without user inputs

Interpolation

```bash
# interpolation

export PYTHONPATH="${PYTHONPATH}:/workspace/code"
export TRAIN_FILE=../data/datasets/debug_data/train.txt
export TEST_FILE=../data/datasets/debug_data/test.txt
export GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python examples/big_ae/run_latent_generation.py \
    --dataset Debug \
    --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
    --output_dir=../output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
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
    --checkpoint_dir=../output/LM/Snli/768/philly_vae_snli_b0.5_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
    --output_dir=../output/LM/Snli/768/philly_vae_snli_b0.5_d5_r00.5_ra0.25_length_weighted/checkpoint-31250 \
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

Please see the scripts I used to run the evaluation at [code/scripts/scripts_local/eval_optimus_latent_space.sh](../code/scripts/scripts_local/eval_optimus_latent_space.sh). Here are some results you can see from the model:



#### When beta changes from 0 to 1, the reconstruction quality become worse

Reconstruction (beta = 0.0)
```
a football coach putting his arm on one of his player's shoulder.  
 a football player putting his arm on one of his coach's shoulder.

 a girl wearing a white shirt and blue jeans jumping off a rock into the sand.  
 a girl wearing a white shirt and blue jeans jumping off a rock into the sand.

 a group of cheerleaders are making a human pyramid at a basketball game.  
 a group of cheerleaders are making a human pyramid at a basketball game.

 a man is looking over things that are in a local market.  
 a man is looking over things that are in a local market.

 a woman is riding a moped on a street with many other people riding mopeds behind her, while streamers and banners hang in the trees overhead.  
 a woman is riding a moped on a street with large trees and other speakers as she rides in front of a turntable carrying camels in the background.

 group of people in the wilderness packing boxes full of food.  
 group of people in the wilderness packing boxes full of food.

 man tries to impress girl by diving into the water.  
 man tries to impress girl by diving into the water.

 people in the middle of city street surrounded by large buildings.  
 people in the middle of city street surrounded by large buildings.

 there are girls that are on the ice practicing their ice dancing.  
 there are girls that are on the ice practicing their ice dancing.

 there are two young boys with shovels who are outside and are digging in the dirt.  
 there are two young boys with shovels and are outside being chased by the dirt.

 two men in blue one standing the other hanging onto the window of a tram car just looking out.  
 two men in blue standing over the window one of them wearing a pink bodysuit riding the other down the street.

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




### When beta changes from 0 to 1, similar interpolation quality are observed:

Interpolation (beta = 0.0)
```
0 
 a woman is riding a moped on a street with large trees and other speakers as she rides in front of a turntable carrying camels in the background.
1 
 a woman is riding a moped on the street with several other people riding mopeds and lights behind him, while passersby in the background draw pictures.
2 
 a woman riding a scooter is riding on two large streets behind her, one with a woman singing in the background behind them.
3 
 a woman on a pony is riding over the street holding several mopeds and wagons in front of a large, white painted building as other people are watching.
4 
 a man with two ponytails riding on the street is riding down an empty stage as another person in a white hooded jacket watches.
5 
 one man in a blue tuxedo is riding over the street holding two people on it as they ride a light brown horse.
6 
 one man in a blue hoodie riding a cart is standing over the street while others view it on the other side.
7 
 two men on one side of the street holding a blue balloon as they ride wagons moving past the building.
8 
 two men in blue sitting on the roof of a car that is blowing up another one leaning very close.
9 
 two men in yellow one standing on the window holding a blue car trying to ride it down the street.
10 
 two men in the blue one window holding onto the car are jumping over another man walking down the street.
```

Interpolation (beta = 0.5)
```
0 
 a woman is riding a moped on a street with many other people behind her, as well as small banners and horns riding in the background.
1 
 a woman is riding a moped with several people on it behind her, riding a straw pole and streets in the background.
2 
 a woman riding a trolley is riding on a street with many people in front of them, as well as ripples surrounding it.
3 
 a woman riding a moped has two others standing in the street on side a bus as they weave, blowing bubbles on it.
4 
 one man riding a black stroller is riding on the street beside a man with painted windows and others populating in the background.
5 
 one man riding a pink bus is standing on the street behind another man making wheelie figures and the window in between them.
6 
 two men in a blue hoodie standing on one of the cars drives past people hanging a wicker window of the street.
7 
 two men in blue holding a wheelie standing on the street that are both winking into the windows next to them.
8 
 two men in pink standing on the street one of which is pulling a blue parasol to window it.
9 
 two men in the blue one person standing on a car leaning over it window dreaming of the other floating.
10 
 two men in blue that was standing next to the window holding one wheelie riding a black bike down the street.

```

Interpolation (beta = 1.0)
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
