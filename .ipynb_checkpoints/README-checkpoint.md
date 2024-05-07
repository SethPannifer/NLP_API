# NLP_API
nlp api yay!

## Docs for CICD
Run sudo sh CICD.sh for updating code

Run sudo sh train.sh for updating model

## Docs for API

/ Root API\
/test Test API\
/run Inference API\
/Hello Easter EGG

## Docker
You can find and exceute the api immedicatlye through Docker

sudo docker pull dockjag/fastnlp:latest
### With GPU
sudo docker run -it --gpus all -p 80:80 dockjag/fastnlp:latest
### On CPU
sudo docker run -it -p 80:80 dockjag/fastnlp:latest
