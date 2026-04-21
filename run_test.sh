pip install -r distributed/requirements-gpu.txt

bash distributed/serve_model.sh \
    /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
    0.0.0.0 8000


cd /mgfs/shared/Group_GY/wenchao/shhh/SWE-bench
python -m distributed.serve_model \
    --model_path /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
    --served_model_name Qwen3.5-9B \
    --host 0.0.0.0 --port 8000 \
    --data_parallel_size 8 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 65536 \
    --dtype bfloat16




pip install -e . && pip install -e ".[datasets]"
pip install -r distributed/requirements-cpu.txt

python -m distributed.check_connection \
    --base_url http://<gpu-host>:8000/v1 --model Qwen3.5-9B

python -m distributed.run_api_remote \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite_oracle \
    --split test \
    --model_name_or_path Qwen3.5-9B \
    --base_url http://<gpu-host>:8000/v1 \
    --output_dir ./outputs --chat

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path ./outputs/Qwen3.5-9B__...__test.jsonl \
    --max_workers 8 \
    --run_id qwen35-9b-lite