export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME="<PATH_TO_GEMMA_TOKENIZOR>"

uv run scripts/run_pi_server.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config=<TRAIN_CONFIG_NAME> \
    --policy.dir=<INFER_CHECKPOINT_DIR> \ 


