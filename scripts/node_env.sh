# Source before any node command: keeps every cache inside the project dir
# so cleanup is one rm -rf and nothing pollutes the shared home or /tmp.
export NLP_ROOT="${NLP_ROOT:-$HOME/nlp}"
export HF_HOME="$NLP_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$NLP_ROOT/.cache"
export PIP_CACHE_DIR="$NLP_ROOT/.cache/pip"
export TORCH_HOME="$NLP_ROOT/.cache/torch"
export MPLCONFIGDIR="$NLP_ROOT/.cache/mpl"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
[ -d "$NLP_ROOT/.venv" ] && . "$NLP_ROOT/.venv/bin/activate"
