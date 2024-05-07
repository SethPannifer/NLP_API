export JOB_NAME="ml-project1"
export IMAGE="dockjag/fastnlp"
export TAG="latest"
export PYTHON_ENV="development"
export API_PORT=80
export WORKERS=10
export TIMEOUT=300
export LOG_FOLDER=/var/log/ml-project1

# stop running container with same job name, if any
if [ "$(docker ps -a | grep $JOB_NAME)" ]; then
  docker stop ${JOB_NAME} && docker rm ${JOB_NAME}
fi

echo ${IMAGE}:${TAG}

# Create log folder if not exists
if [ ! -d ${LOG_FOLDER} ]; then
     mkdir ${LOG_FOLDER}
fi


# docker image build --compress -t ${IMAGE}:${TAG} .

# docker push ${IMAGE}:${TAG}

docker run -it \
  --rm \
  --gpus all \
  -e "WORKERS=${WORKERS}" \
  -e "PYTHON_ENV=${PYTHON_ENV}" \
  --name="${JOB_NAME}" \
  ${IMAGE}:${TAG} \
  python app/Train_Model.py --s 100 --n Test --d surrey-nlp/PLOD-CW
