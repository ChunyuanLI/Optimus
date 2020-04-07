# Optimus: the first pre-trained Big VAE language model <img src="doc/figs/logo_optimus.png" width="80">  


This repository contains source code necessary to reproduce the results presented in the paper [Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space](https://arxiv.org/).


|<img src="doc/figs/optimus_scheme.png" width="350"> | <img src="doc/figs/headfig_optimus.png" width="800"> 
|-------------------------|:-------------------------:|
| The network architecture of Optimus: encoder for representation learning and decoder for generation | Sentences are organized and manipulated in a pre-trained compact and smooth latent space 


For more on this project, see the [Microsoft Research Blog post](https://www.microsoft.com/en-us/research/blog/).


### [Code Cleaning in Progress, April 7, 2020]

## Contents
There are four steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#dependencies)
2. [Prepare datasets](#prepare-datasets)
3. [Model training](#Model-training)
    1. Pre-training on setences in Wikipedia
    2. Languange Modeling
    3. Guided Language Generation
    4. Low-resource Language Understanding
4. [Collect and plot results](#collect-and-plot-results)


## Dependencies

Pull docker from Docker Hub at: chunyl/pytorch-transformers:v2

Edit the project path to the absolute path on your computer by changing the "SCRIPTPATH" in "code/scripts/scripts_docker/run_docker.sh"

CD into the directory "code", and run docker

    sh scripts/scripts_docker/run_docker.sh
  

## Prepare Datasets

Please the data preparation at links:

## Model Training

**1. Pre-training on setences in Wikipedia**

**2. Languange Modeling**

**3. Guided Language Generation**

**4. Low-resource Language Understanding**

## Collect and Plot Results

Once the networks are trained and the results are saved, we extracted key results using Python script. The results can be plotted using the included IPython notebook `plots/main_plots.ipynb`.
Start the IPython Notebook server:

```
$ cd plots
$ ipython notebook
```

Select the `main_plots.ipynb` notebook and execute the included
code. Note that without modification, we have copyed our extracted results into the notebook, and script will output figures in the paper. If you've run your own training and wish to plot results, you'll have to organize your results in the same format instead.


## Questions?

Please drop me ([Chunyuan](http://chunyuan.li/)) a line if you have any questions.


```
@inproceedings{li2020_Optimus,
  title={Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space},
  author={Li, Chunyuan and Gao, Xiang and Li, Yuan and Li, Xiujun and Peng, Baolin and Zhang, Yizhe and Gao, Jianfeng},
  booktitle={arXiv},
  year={2020}
}
```


