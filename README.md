# NLP_API
nlp api yay!

## Docs for CICD
Run sudo sh build.sh for building Docker

Run sudo sh CICD.sh for updating code

Run sudo sh train.sh for updating model

## Docs for API

/ Root API\
/test Test API\
/run Inference API\
/Hello Easter EGG

## Docker
You can find and exceute the api immedicatlye through Docker

### Download the docker image
sudo docker pull dockjag/fastnlp:latest

### On GPU
docker run -it --gpus all -p 80:80 dockjag/fastnlp:latest

OR

sh run.sh

### On CPU
docker run -it -p 80:80 dockjag/fastnlp:latest

### Train 
sh train.sh
