MODEL_PATH="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
HOST="localhost"
PORT="40220"



bash distributed/serve_model.sh \
    ${MODEL_PATH} \
    ${HOST} ${PORT}