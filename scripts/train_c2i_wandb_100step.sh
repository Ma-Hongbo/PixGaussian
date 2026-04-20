#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export WANDB_API_KEY=...
#   bash scripts/train_c2i_wandb_100step.sh fit
#   bash scripts/train_c2i_wandb_100step.sh fit datasets/imagenette2-320/train my-run pixnerd-c2i
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_c2i_wandb_100step.sh fit
#   bash scripts/train_c2i_wandb_100step.sh fit ... --save_ckpt /path/to/checkpoints
#
# Args:
#   $1 MODE        : fit | predict (default: fit)
#   $2 DATA_ROOT   : train imagefolder root (default: datasets/imagenette2-320/train)
#   $3 RUN_NAME    : wandb run name (default: c2i-<timestamp>)
#   $4 PROJECT     : wandb project (default: pixnerd-c2i)
#   $5 CONFIG      : config path (default: configs_c2i/pix256_c2i_wandb_100step.yaml)
#   $6 DINO_PATH   : optional local torch.hub DINOv2 path; if omitted, use $DINO_WEIGHT_PATH
#   $7... EXTRA    : extra LightningCLI overrides
#                    script-specific option:
#                    --save_ckpt <path> | --save-ckpt <path>
#                    --ckpt_dir <path>  | --ckpt-dir <path>

MODE="${1:-fit}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  sed -n '1,50p' "$0"
  exit 0
fi
DATA_ROOT="${2:-datasets/imagenette2-320/train}"
RUN_NAME="${3:-c2i-$(date +%y%m%d-%H%M%S)}"
PROJECT="${4:-pixnerd-c2i}"
CONFIG="${5:-configs_c2i/pix256_c2i_wandb_100step.yaml}"
RAW_ARG6="${6:-}"
DINO_PATH="${DINO_WEIGHT_PATH:-}"
SHIFT_N=5
if [[ -n "${RAW_ARG6}" && "${RAW_ARG6}" != --* ]]; then
  DINO_PATH="${RAW_ARG6}"
  SHIFT_N=6
fi
for ((i=0; i<SHIFT_N && $#>0; i++)); do
  shift
done

CKPT_DIR="${PIXNERD_CKPT_DIR:-}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save_ckpt|--save-ckpt|--ckpt_dir|--ckpt-dir)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] $1 requires a path argument"
        exit 1
      fi
      CKPT_DIR="$2"
      shift 2
      ;;
    --save_ckpt=*|--save-ckpt=*|--ckpt_dir=*|--ckpt-dir=*)
      CKPT_DIR="${1#*=}"
      shift 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

has_tags_exp_override=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--tags.exp" || "${arg}" == --tags.exp=* ]]; then
    has_tags_exp_override=true
    break
  fi
done

if [[ "${MODE}" != "fit" && "${MODE}" != "predict" ]]; then
  echo "[ERROR] MODE must be fit or predict, got: ${MODE}"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}"
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[WARN] WANDB_API_KEY is empty. Wandb may run in offline/disabled mode."
fi

if [[ "${MODE}" == "fit" && ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}"
  echo "        Run: bash scripts/download_c2i_dataset.sh"
  exit 1
fi

if [[ -z "${DINO_PATH}" ]]; then
  echo "[ERROR] DINO_PATH is empty."
  echo "        Pass arg #6 or set env DINO_WEIGHT_PATH."
  echo "        Example:"
  echo "        export DINO_WEIGHT_PATH=/path/to/facebookresearch_dinov2_main/dinov2_vitb14"
  exit 1
fi

DINO_REPO_DIR="$(dirname "${DINO_PATH}")"
DINO_HUB_ENTRY="$(basename "${DINO_PATH}")"
if [[ ! -d "${DINO_REPO_DIR}" || ! -f "${DINO_REPO_DIR}/hubconf.py" ]]; then
  echo "[ERROR] Invalid DINO_PATH: ${DINO_PATH}"
  echo "        Expected format: /path/to/facebookresearch_dinov2_main/dinov2_vitb14"
  echo "        The repo directory must exist and contain hubconf.py:"
  echo "        ${DINO_REPO_DIR}/hubconf.py"
  exit 1
fi

if [[ ! -e "${DINO_PATH}" ]]; then
  echo "[WARN] DINO hub entry path does not exist on disk: ${DINO_PATH}"
  echo "       Continuing because the parent repo looks valid."
  echo "       torch.hub will load entry: ${DINO_HUB_ENTRY}"
fi

echo "[INFO] Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[INFO] Mode: ${MODE}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Data root: ${DATA_ROOT}"
echo "[INFO] DINO path: ${DINO_PATH}"
echo "[INFO] DINO repo/entry: ${DINO_REPO_DIR} / ${DINO_HUB_ENTRY}"
echo "[INFO] Wandb project/run: ${PROJECT}/${RUN_NAME}"
TAG_EXP_SANITIZED="$(echo "${RUN_NAME}" | sed 's#[/: ]#_#g')"
echo "[INFO] tags.exp: ${TAG_EXP_SANITIZED}"
if [[ -n "${CKPT_DIR}" ]]; then
  echo "[INFO] Checkpoint dir: ${CKPT_DIR}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "[INFO] Extra args: ${EXTRA_ARGS[*]:-(none)}"

if [[ -n "${CKPT_DIR}" ]]; then
  export PIXNERD_CKPT_DIR="${CKPT_DIR}"
fi

cmd=(
  python main.py "${MODE}" -c "${CONFIG}"
  --data.train_dataset.init_args.root "${DATA_ROOT}"
  --model.diffusion_trainer.init_args.encoder.init_args.weight_path "${DINO_PATH}"
  --trainer.logger.init_args.project "${PROJECT}"
  --trainer.logger.init_args.name "${RUN_NAME}"
)

if [[ "${has_tags_exp_override}" == "false" ]]; then
  cmd+=(--tags.exp "${TAG_EXP_SANITIZED}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Running: ${cmd[*]}"
"${cmd[@]}"
