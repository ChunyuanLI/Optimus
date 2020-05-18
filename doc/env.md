# Set up the Environment

Pull docker from Docker Hub at: `chunyl/pytorch-transformers:v2`, and run it using the following script:


```
SCRIPTPATH="/home/chunyl/azure_mounts/optimus_azure"
IMAGE=chunyl/pytorch-transformers:v2

docker run \
--runtime=nvidia \
-it --rm \
--net host \
--volume $SCRIPTPATH:/workspace \
--interactive --tty $IMAGE /bin/bash

```


There is an example at `code/scripts/scripts_docker/run_docker.sh`. Please edit the project path to the absolute path on your computer by changing the "SCRIPTPATH", then run the docker at the the directory "code":

    sh scripts/scripts_docker/run_docker.sh
