
#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# SSLAM ESC-50 Fine-tuning
# Original SSLAM Fairseq framework
#
# Usage:
#   bash scripts/train_esc50.sh 1 --dry-run
#   bash scripts/train_esc50.sh 1
#
# Required:
#   FAIRSEQ_ROOT
#   PRETRAINED_CKPT
#
# Optional:
#   MANIFEST_ROOT
#   BATCH_SIZE
#   UPDATE_FREQ
#   MAX_UPDATE
#   WARMUP_UPDATES
#   PYTHON_BIN
# ==========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FOLD="${1:-}"
MODE="${2:-}"

if [[ ! "$FOLD" =~ ^[1-5]$ ]]; then
    echo "Usage: bash scripts/train_esc50.sh <1-5> [--dry-run]"
    exit 1
fi

if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
    echo "Error: unsupported option $MODE"
    exit 1
fi

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"

MANIFEST_ROOT="${MANIFEST_ROOT:-${PROJECT_ROOT}/datasets/manifests/ESC50}"

PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_DIR="${PROJECT_ROOT}/configs"
CONFIG_NAME="sslam_esc50"

USER_DIR="${PROJECT_ROOT}/models/SSLAM/SSLAM_Finetune/SSLAM_Ft_AS20K"

DATA_DIR="${MANIFEST_ROOT}/fold_${FOLD}"

OUTPUT_DIR="${PROJECT_ROOT}/results/sslam_esc50/fold_${FOLD}"

LOG_DIR="${PROJECT_ROOT}/experiments/sslam_esc50/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ----------------------------------------------------------
# Training parameters
# ----------------------------------------------------------

BATCH_SIZE="${BATCH_SIZE:-48}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"

MAX_UPDATE="${MAX_UPDATE:-8000}"
WARMUP_UPDATES="${WARMUP_UPDATES:-800}"

LEARNING_RATE="0.0001"

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

if [[ -z "$FAIRSEQ_ROOT" ]]; then
    echo "ERROR: FAIRSEQ_ROOT is not set."
    exit 1
fi

FAIRSEQ_ROOT="$(realpath "$FAIRSEQ_ROOT")"

if [[ ! -f "$FAIRSEQ_ROOT/fairseq_cli/hydra_train.py" ]]; then
    echo "ERROR: Fairseq training entry point not found."
    echo "$FAIRSEQ_ROOT/fairseq_cli/hydra_train.py"
    exit 1
fi

if [[ -z "$PRETRAINED_CKPT" || ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: PRETRAINED_CKPT is missing or invalid."
    exit 1
fi

PRETRAINED_CKPT="$(realpath "$PRETRAINED_CKPT")"

if [[ ! -f "$CONFIG_DIR/${CONFIG_NAME}.yaml" ]]; then
    echo "ERROR: Configuration file not found."
    exit 1
fi

if [[ ! -d "$USER_DIR/tasks" || ! -d "$USER_DIR/models" ]]; then
    echo "ERROR: SSLAM tasks/models directories not found."
    exit 1
fi

for file in \
    train.tsv \
    train.lbl \
    eval.tsv \
    eval.lbl \
    label_descriptors.csv
do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        echo "ERROR: Missing $DATA_DIR/$file"
        exit 1
    fi
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi

# ----------------------------------------------------------
# Log information
# ----------------------------------------------------------

echo "============================================"
echo "SSLAM ESC-50 Fine-tuning"
echo "============================================"

echo "Project:       $PROJECT_ROOT"
echo "Fold:          $FOLD"
echo "Fairseq:       $FAIRSEQ_ROOT"
echo "User module:   $USER_DIR"
echo "Dataset:       $DATA_DIR"
echo "Checkpoint:    $PRETRAINED_CKPT"

echo "Batch size:    $BATCH_SIZE"
echo "Update freq:   $UPDATE_FREQ"
echo "Max updates:   $MAX_UPDATE"
echo "Warmup:        $WARMUP_UPDATES"
echo "Learning rate: $LEARNING_RATE"

echo "Output:        $OUTPUT_DIR"
echo "============================================"

# ----------------------------------------------------------
# Fairseq / Hydra command
# ----------------------------------------------------------

export PYTHONPATH="${FAIRSEQ_ROOT}:${PYTHONPATH:-}"

TRAIN_CMD=(
    "$PYTHON_BIN"
    "$FAIRSEQ_ROOT/fairseq_cli/hydra_train.py"
    -m

    --config-dir "$CONFIG_DIR"
    --config-name "$CONFIG_NAME"

    "common.user_dir=$USER_DIR"

    "hydra.sweep.dir=$OUTPUT_DIR/hydra"

    "checkpoint.save_dir=$OUTPUT_DIR/checkpoints"
    "checkpoint.best_checkpoint_metric=accuracy"
    "checkpoint.maximize_best_checkpoint_metric=true"
    "checkpoint.no_save=false"

    "task.data=$DATA_DIR"
    "task.esc50_eval=true"

    "model.model_path=$PRETRAINED_CKPT"
    "model.esc50_eval=true"
    "model.num_classes=50"
    "model.mixup=0.0"
    "model.mixup_prob=0.0"

    "dataset.batch_size=$BATCH_SIZE"

    "optimization.update_freq=[$UPDATE_FREQ]"
    "optimization.max_update=$MAX_UPDATE"
    "optimization.lr=[$LEARNING_RATE]"

    "optimizer.groups.default.lr_float=$LEARNING_RATE"
    "optimizer.groups.default.lr_scheduler.warmup_updates=$WARMUP_UPDATES"
)

# ----------------------------------------------------------
# Dry run
# ----------------------------------------------------------

if [[ "$MODE" == "--dry-run" ]]; then
    echo
    echo "DRY RUN: training will not start."
    echo
    printf '%q ' "${TRAIN_CMD[@]}"
    echo
    exit 0
fi

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

echo
echo "Starting SSLAM ESC-50 Fold $FOLD"
echo

LOG_FILE="$LOG_DIR/fold_${FOLD}.log"

"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo
echo "Training process completed for Fold $FOLD"
echo "Log saved at: $LOG_FILE"
