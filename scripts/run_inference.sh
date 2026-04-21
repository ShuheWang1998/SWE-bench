DATASET_NAME_OR_PATH="princeton-nlp/SWE-bench_oracle"
SPLIT="test"
MODEL_NAME_OR_PATH="Qwen3.5-9B"
BASE_URL="http://localhost:40220/v1"
OUTPUT_DIR="/home/wangshuhe/results/swe_qwen35_9b_outputs"
TOKENIZER="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
MAX_NEW_TOKENS="0"


python -m distributed.run_api_remote \
    --dataset_name_or_path ${DATASET_NAME_OR_PATH} --split ${SPLIT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --base_url ${BASE_URL} --output_dir ${OUTPUT_DIR} \
    --tokenizer ${TOKENIZER} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chat