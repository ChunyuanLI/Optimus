## Set up Environment

Pull docker from Docker Hub at: chunyl/pytorch-transformers:v2

Edit the project path to the absolute path on your computer by changing the "SCRIPTPATH" in "code/scripts/scripts_docker/run_docker.sh"

CD into the directory "code", and run docker

    sh scripts/scripts_docker/run_docker.sh
    
    

  
## Fine-tune Language Models

    sh scripts/scripts_local/run_ft_lm_vae_optimus.sh
    
    


## Play with the latent space


    sh scripts/scripts_local/eval_optimus_latent_space.sh
    
    
