## Set up Environment

Pull docker from Docker Hub at: chunyl/pytorch-transformers:v2

Edit the project path to the absolute path on your computer by changing the "SCRIPTPATH" in "code/scripts/scripts_docker/run_docker.sh"

CD into the directory "code", and run docker

    sh scripts/scripts_docker/run_docker.sh
    
    

  
## Fine-tune Language Models

    sh scripts/scripts_local/run_ft_lm_vae_optimus.sh
    
    
The main training script is [`run_lm_vae_training.py`](./examples/big_ae/run_lm_vae_training.py) and conducts the fine-tuning loop, taking the following options (among others) as arguments:

- The `two positional arguments` specify the paths for training and validation sets (two _hdf5_ files), respectively; these arguments are required.
- `--checkpoint_dir`: the folder that the pre-trained Optimus is saved.
- `--gloabl_step_eval`: it specifies the checkpoint (the steps that Optimus is trained).
- `--num_train_epochs`: number of training epochs (type=int); default 1.
- `--dim_target_kl`:   the hyper-paramter used in dimension-wise thresholding used in fine-tuning(type=float); default 0.5.
- `--beta`:   the maximum beta value used in cyclical annealing schedule used in fine-tuning(type=float); default 1.0.
- `--ratio_zero`:   the proportion of beta=0 in one period for fine-tuning(type=float); default 0.5
- `--ratio_increase`:  the proportion of beta that increases from 0 to the maximum value in one period in cyclical annealing schedule used in fine-tuning(type=float); default 0.25.


For more options, please see [`standard_parser.py`](./intrinsic_dim/standard_parser.py) and [`train.py`](./intrinsic_dim/train.py), or just run `./train.py -h`.


## Play with the latent space


    sh scripts/scripts_local/eval_optimus_latent_space.sh
    
    
