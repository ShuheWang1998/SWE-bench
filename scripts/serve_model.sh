MODEL_PATH="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"
HOST="0.0.0.0"
PORT="8000"
TENSOR_PARALLEL_SIZE="1"
DATA_PARALLEL_SIZE="8"
GPU_MEMORY_UTILIZATION="0.9"
MAX_MODEL_LEN="65536"
DTYPE="bfloat16"



python -m distributed.serve_model \
    --model_path ${MODEL_PATH} \
    --served_model_name ${SERVED_MODEL_NAME} \
    --host ${HOST} --port ${PORT} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --data_parallel_size ${DATA_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --max_model_len ${MAX_MODEL_LEN} \
    --dtype ${DTYPE}