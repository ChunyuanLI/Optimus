# Optimus


Download code:


Outline:

code: code and scripts to run the project in different computing environments
output: checkpoints to save models and intermediate results (including our pre-trained models).
data: various datasets and pre-processed versions.


### Download Pre-trained model

wget https://textae.file.core.windows.net/textae/textae.tar.gz?st=2019-11-23T00%3A21%3A21Z&se=2019-11-24T00%3A21%3A21Z&sp=rl&sv=2018-03-28&sr=f&sig=4jWG8Qks7lX8n0%2BLMmKNhnEpiV2CL1NqO2fwbzTEc1M%3D

Put the downloaded folder into "output".

### Download and run Docker

Pull docker from Docker Hub at: chunyl/pytorch-transformers:v2

Edit the project path to the absolute path on your computer by changing the "SCRIPTPATH" in "code/scripts/scripts_docker/run_docker.sh"

CD into the directory "code", and run docker "sh scripts/scripts_docker/run_docker.sh" 

### Download GLUE datasets

Following HuggingFace repo, before running anyone of GLUE tasks, you should download the GLUE data by running this script and unpack it to directory data/datasets/glue_data/.

(You may also install the additional packages required: pip install -r ./examples/requirements.txt)

### Run/Compare feature-based classification

Let's take QNLI for example (104k training instances, 5.4k validation instances). You should be able to see the following accuracy by run the scripts:
 
Optimus: acc = 0.7062

sh scripts/scripts_local/run_vae_glue.sh

BERT: acc = 0.6624

sh scripts/scripts_local/run_bert_glue.sh


