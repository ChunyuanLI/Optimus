SCRIPTPATH="/home/chunyl/research/project/optimus"
IMAGE=chunyl/pytorch-transformers:v2

docker run \
--runtime=nvidia \
-it --rm \
--net host \
--volume $SCRIPTPATH:/workspace \
--interactive --tty $IMAGE /bin/bash


