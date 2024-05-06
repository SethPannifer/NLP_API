export JOB_NAME="ml-project1"
export IMAGE="dockjag/fnlp"
export TAG="latest"
export PYTHON_ENV="development"
export API_PORT=8080
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


docker image build --compress -t ${IMAGE}:${TAG} .

# docker push ${IMAGE}:${TAG}

docker run -d \
  --rm \
  -p ${API_PORT}:80 \
  -e "WORKERS=${WORKERS}" \
  -e "TIMEOUT=${TIMEOUT}" \
  -e "PYTHON_ENV=${PYTHON_ENV}" \
  --name="${JOB_NAME}" \
  ${IMAGE}:${TAG}

