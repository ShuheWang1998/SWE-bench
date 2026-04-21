MODEL_PATH="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
HOST="localhost"
PORT="40220"



python -m distributed.check_connection \
  --base_url http://${HOST}:${PORT}/v1 --model Qwen3.5-9B