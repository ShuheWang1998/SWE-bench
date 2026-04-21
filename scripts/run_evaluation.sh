PREDICTIONS_PATH="/home/wangshuhe/results/swe_qwen35_9b_outputs/Qwen3.5-9B__princeton-nlp__SWE-bench_oracle__test.jsonl"
DATASET_NAME_OR_PATH="princeton-nlp/SWE-bench_oracle"
SPLIT="test"
MODEL_NAME_OR_PATH="Qwen3.5-9B"
MAX_WORKERS="8"
RUN_ID="qwen35-9b-max-tokens-200"


python -m swebench.harness.run_evaluation \
    --dataset_name ${DATASET_NAME_OR_PATH} \
    --predictions_path ${PREDICTIONS_PATH} \
    --max_workers ${MAX_WORKERS} \
    --run_id ${RUN_ID}