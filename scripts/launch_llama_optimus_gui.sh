#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_LLAMA_BIN="${LLAMA_BIN:-/home/macko/llm/llama.cpp/build/bin}"
DEFAULT_MODEL_DIR="${MODEL_DIR:-/home/macko/models}"
DEFAULT_PORT="${OPTIMUS_SERVER_PORT:-8080}"
DEFAULT_TRIALS="${OPTIMUS_TRIALS:-}"
DEFAULT_REPEAT="${OPTIMUS_REPEAT:-}"
DEFAULT_TOKENS="${OPTIMUS_N_TOKENS:-64}"
DEFAULT_WARMUP_RUNS="${OPTIMUS_WARMUP_RUNS:-6}"
DEFAULT_WARMUP_TOKENS="${OPTIMUS_WARMUP_TOKENS:-64}"
DEFAULT_PROFILE="${OPTIMUS_PROFILE:-gpu-first}"
LOG_DIR="${OPTIMUS_LOG_DIR:-$REPO_ROOT/logs}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        zenity --error --title="TEST_llama-optimus" --text="Missing required command: $1"
        exit 1
    fi
}

shell_quote() {
    printf '%q' "$1"
}

port_in_use() {
    ss -ltn "sport = :$1" | tail -n +2 | grep -q .
}

require_command zenity
require_command gnome-terminal
require_command ss

mkdir -p "$LOG_DIR"

if [[ ! -x "$DEFAULT_LLAMA_BIN/llama-server" ]]; then
    zenity --error --title="TEST_llama-optimus" \
        --text="llama-server not found in:\n$DEFAULT_LLAMA_BIN"
    exit 1
fi

if [[ ! -x "$DEFAULT_LLAMA_BIN/llama-bench" ]]; then
    zenity --error --title="TEST_llama-optimus" \
        --text="llama-bench not found in:\n$DEFAULT_LLAMA_BIN"
    exit 1
fi

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
    zenity --error --title="TEST_llama-optimus" \
        --text="Python virtualenv missing:\n$REPO_ROOT/.venv/bin/python"
    exit 1
fi

MODEL_PATH="$(zenity --file-selection \
    --title="Choose GGUF model" \
    --filename="$DEFAULT_MODEL_DIR/")"

if [[ -z "${MODEL_PATH:-}" ]]; then
    exit 0
fi

MODEL_CONTEXT="$("$REPO_ROOT/.venv/bin/python" - <<'PY' "$REPO_ROOT" "$MODEL_PATH" "$DEFAULT_LLAMA_BIN"
import sys
repo_root, model_path, llama_bin = sys.argv[1:4]
sys.path.insert(0, f"{repo_root}/src")
from llama_optimus.model_arch import detect_max_context

context = detect_max_context(model_path, llama_bin)
print("" if context is None else context)
PY
)"

CTX_VALUES=(2048 8192 16384 32768 65536 131072 262144 524288 1048576)
CTX_LABELS=("2K" "8K" "16K" "32K" "65K" "131K" "262K" "512K" "1M")
AVAILABLE_CTX_VALUES=()
AVAILABLE_CTX_LABELS=()
DEFAULT_CTX=""

for i in "${!CTX_VALUES[@]}"; do
    value="${CTX_VALUES[$i]}"
    label="${CTX_LABELS[$i]}"
    if [[ -z "$MODEL_CONTEXT" || "$value" -le "$MODEL_CONTEXT" ]]; then
        AVAILABLE_CTX_VALUES+=("$value")
        AVAILABLE_CTX_LABELS+=("$label")
        DEFAULT_CTX="$value"
    fi
done

if [[ ${#AVAILABLE_CTX_VALUES[@]} -eq 0 ]]; then
    AVAILABLE_CTX_VALUES=("8192")
    AVAILABLE_CTX_LABELS=("8K")
    DEFAULT_CTX=8192
fi

CTX_CHOICES=()
for i in "${!AVAILABLE_CTX_VALUES[@]}"; do
    pick="FALSE"
    if [[ "${AVAILABLE_CTX_VALUES[$i]}" == "$DEFAULT_CTX" ]]; then
        pick="TRUE"
    fi
    CTX_CHOICES+=("$pick" "${AVAILABLE_CTX_VALUES[$i]}" "${AVAILABLE_CTX_LABELS[$i]}")
done

CTX_PROMPT="Select llama.cpp ctx-size"
if [[ -n "$MODEL_CONTEXT" ]]; then
    CTX_PROMPT+="\nModel max context detected: $MODEL_CONTEXT"
else
    CTX_PROMPT+="\nModel max context could not be detected"
fi

CTX_SIZE="$(zenity --list \
    --title="Choose context size" \
    --text="$CTX_PROMPT" \
    --radiolist \
    --column="Pick" --column="Tokens" --column="Label" \
    "${CTX_CHOICES[@]}" \
    --height=360 --width=520)"

if [[ -z "${CTX_SIZE:-}" ]]; then
    CTX_SIZE="$DEFAULT_CTX"
fi

PORT="$(zenity --entry \
    --title="Choose local port" \
    --text="Enter the local port for llama-server" \
    --entry-text="$DEFAULT_PORT")"

if [[ -z "${PORT:-}" ]]; then
    exit 0
fi

PROFILE="$(zenity --list \
    --title="Choose tuning profile" \
    --text="Select the optimization strategy" \
    --radiolist \
    --column="Pick" --column="Profile" --column="Meaning" \
    TRUE "gpu-first" "Prefer keeping work on GPU and avoid aggressive offload" \
    FALSE "balanced" "General-purpose compromise" \
    FALSE "throughput" "Push for raw tok/sec" \
    FALSE "economic" "Prefer efficiency and lighter memory use" \
    --height=320 --width=760)"

if [[ -z "${PROFILE:-}" ]]; then
    PROFILE="$DEFAULT_PROFILE"
fi

if [[ -z "$DEFAULT_TRIALS" ]]; then
    if [[ "$PROFILE" == "throughput" ]]; then
        DEFAULT_TRIALS=30
    else
        DEFAULT_TRIALS=15
    fi
fi

if [[ -z "$DEFAULT_REPEAT" ]]; then
    if [[ "$PROFILE" == "throughput" ]]; then
        DEFAULT_REPEAT=2
    else
        DEFAULT_REPEAT=2
    fi
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    zenity --error --title="TEST_llama-optimus" \
        --text="Invalid port: $PORT"
    exit 1
fi

if [[ ! "$CTX_SIZE" =~ ^[0-9]+$ ]] || (( CTX_SIZE < 1 )); then
    zenity --error --title="TEST_llama-optimus" \
        --text="Invalid ctx-size: $CTX_SIZE"
    exit 1
fi

if port_in_use "$PORT"; then
    zenity --error --title="TEST_llama-optimus" \
        --text="Port $PORT is already in use."
    exit 1
fi

LD_LIBRARY_PREFIX="$DEFAULT_LLAMA_BIN"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    LD_LIBRARY_PREFIX="$DEFAULT_LLAMA_BIN:$LD_LIBRARY_PATH"
fi

OPTIMIZER_CMD=(
    env
    "PYTHONPATH=$REPO_ROOT/src"
    "LD_LIBRARY_PATH=$LD_LIBRARY_PREFIX"
    "$REPO_ROOT/.venv/bin/python"
    "$REPO_ROOT/optimus.py"
    --llama-bin "$DEFAULT_LLAMA_BIN"
    --model "$MODEL_PATH"
    --arch auto
    --profile "$PROFILE"
    --trials "$DEFAULT_TRIALS"
    --repeat "$DEFAULT_REPEAT"
    --n-tokens "$DEFAULT_TOKENS"
    --n-warmup-runs "$DEFAULT_WARMUP_RUNS"
    --n-warmup-tokens "$DEFAULT_WARMUP_TOKENS"
)

OPTIMIZER_CMD_STR=""
for arg in "${OPTIMIZER_CMD[@]}"; do
    OPTIMIZER_CMD_STR+=" $(shell_quote "$arg")"
done

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="$LOG_DIR/server-$TIMESTAMP.log"
OPTIMIZER_LOG="$LOG_DIR/optimizer-$TIMESTAMP.log"

gnome-terminal \
    --title="TEST_llama-optimus orchestrator" \
    -- bash -lc "
set -euo pipefail

OPT_LOG=$(shell_quote "$OPTIMIZER_LOG")
SRV_LOG=$(shell_quote "$SERVER_LOG")
PORT=$(shell_quote "$PORT")
LD_PREFIX=$(shell_quote "$LD_LIBRARY_PREFIX")
CTX_SIZE=$(shell_quote "$CTX_SIZE")

echo 'Running optimizer first so large models do not OOM on a naive server start.'
${OPTIMIZER_CMD_STR# } 2>&1 | tee \"$OPT_LOG\"

LAUNCH_CMD=\$(grep -E '^Launch Command:' \"$OPT_LOG\" | tail -n 1 | sed 's/^Launch Command: //')
if [[ -z \"\${LAUNCH_CMD:-}\" ]]; then
    echo
    echo 'Optimizer did not produce a launch command.'
    echo \"Optimizer log: $OPTIMIZER_LOG\"
    echo 'Press Enter to close.'
    read -r
    exit 1
fi

if [[ \" \$LAUNCH_CMD \" != *' --port '* ]]; then
    LAUNCH_CMD+=\" --port $PORT\"
fi
if [[ \" \$LAUNCH_CMD \" != *' --host '* ]]; then
    LAUNCH_CMD+=' --host 0.0.0.0'
fi
if [[ \" \$LAUNCH_CMD \" != *' --ctx-size '* && \" \$LAUNCH_CMD \" != *' -c '* ]]; then
    LAUNCH_CMD+=\" --ctx-size $CTX_SIZE\"
fi

echo
echo 'Starting optimized server:'
echo \"\$LAUNCH_CMD\"
echo
env LD_LIBRARY_PATH=\"$LD_PREFIX\" bash -lc \"\$LAUNCH_CMD\" 2>&1 | tee \"$SRV_LOG\"

echo
echo \"Optimizer log: $OPTIMIZER_LOG\"
echo \"Server log: $SERVER_LOG\"
echo 'Process finished. Press Enter to close.'
read -r"

zenity --info \
    --title="TEST_llama-optimus" \
    --width=420 \
    --text="Optimizer started first, then the optimized server will launch on port $PORT.\n\nProfile: $PROFILE\nctx-size: $CTX_SIZE\nModel:\n$MODEL_PATH\n\nLogs:\n$OPTIMIZER_LOG\n$SERVER_LOG"
