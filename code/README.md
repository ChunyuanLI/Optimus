## Set up Environment

Pull docker from Docker Hub at: chunyl/pytorch-transformers:v2

Edit the project path to the absolute path on your computer by changing the "SCRIPTPATH" in [run_docker.sh](./scripts/scripts_docker/run_docker.sh)

In this directory ("code"), and run docker

    sh scripts/scripts_docker/run_docker.sh
    
    

  
## Fine-tune Language Models

    sh scripts/scripts_local/run_ft_lm_vae_optimus.sh
    
    
The main training script is [`run_lm_vae_training.py`](./examples/big_ae/run_lm_vae_training.py) and conducts the fine-tuning loop, taking the following options (among others) as arguments:

- `--checkpoint_dir`: the folder that the pre-trained Optimus is saved.
- `--gloabl_step_eval`: it specifies the checkpoint (the steps that Optimus is trained).
- `--train_data_file` and `--eval_data_file`: the path for training and testing datasets for the downstream fine-tuning.
- `--dataset`: the dataset for fine-tuning. such as `Penn`
- `--num_train_epochs`: number of training epochs (type=int); default 1.
- `--dim_target_kl`:   the hyper-paramter used in dimension-wise thresholding used in fine-tuning(type=float); default 0.5.
- `--beta`:   the maximum beta value used in cyclical annealing schedule used in fine-tuning(type=float); default 1.0.
- `--ratio_zero`:   the proportion of beta=0 in one period for fine-tuning(type=float); default 0.5
- `--ratio_increase`:  the proportion of beta that increases from 0 to the maximum value in one period in cyclical annealing schedule used in fine-tuning(type=float); default 0.25.


For more options, please see [`run_lm_vae_training.py`](./examples/big_ae/run_lm_vae_training.py) and  see the examples we provided in [`run_ft_lm_vae_optimus.sh`](./scripts/scripts_local/run_ft_lm_vae_optimus.sh), or [more running scripts we used to run the code on a cluster](./scripts/scripts_philly).


## Play with the latent space

    sh scripts/scripts_local/eval_optimus_latent_space.sh
    
The main training script is [`run_latent_generation.py`](./examples/big_ae/run_latent_generation.py) and evaluates the various ways to generate text conditioned on latent vectors, taking the following options (among others) as arguments:

- `--play_mode`:  The current scripts supports two ways to play with the pre-trained VAE models: [`reconstrction`, `interpolation`]
