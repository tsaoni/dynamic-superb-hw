export HF_DATASETS_CACHE=/tmp2/yuling/dataset
export TRANSFORMERS_CACHE=/tmp2/yuling/transformers
export WANDB_CACHE_DIR=/tmp2/yuling/.cache/wandb
export WANDB_CONFIG_DIR=/tmp2/yuling/.config/wandb
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

unset LD_LIBRARY_PATH
python main.py