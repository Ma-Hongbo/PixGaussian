#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Usage:
#   export WANDB_API_KEY=...
#   export DINO_WEIGHT_PATH=/path/to/facebookresearch_dinov2_main/dinov2_vitb14
#   bash scripts/train_c2i_ckpt3000.sh fit
#   bash scripts/train_c2i_ckpt3000.sh fit datasets/imagenette2-320/train my-run pixnerd-c2i
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_c2i_ckpt3000.sh fit
#   bash scripts/train_c2i_ckpt3000.sh fit ... --save_ckpt /path/to/checkpoints
#
# Args:
#   $1 MODE        : fit | predict (default: fit)
#   $2 DATA_ROOT   : train imagefolder root (default: datasets/imagenette2-320/train)
#   $3 RUN_NAME    : wandb run name (default: c2i-<timestamp>)
#   $4 PROJECT     : wandb project (default: pixnerd-c2i)
#   $5 CONFIG      : config path (default: configs_c2i/pix256_c2i_wandb_100step.yaml)
#   $6 DINO_PATH   : optional local torch.hub DINOv2 path; if omitted, use $DINO_WEIGHT_PATH
#   $7... EXTRA    : extra LightningCLI overrides
#                    script-specific options:
#                    --save_ckpt <path> | --save-ckpt <path>
#                    --ckpt_dir <path>  | --ckpt-dir <path>
#                    --save_every_steps <int> | --save-every-steps <int> (default: 3000)

MODE="${1:-fit}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  sed -n '1,70p' "$0"
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
SAVE_EVERY_STEPS="${PIXNERD_SAVE_EVERY_N_TRAIN_STEPS:-3000}"
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
    --save_every_steps|--save-every-steps)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] $1 requires an integer argument"
        exit 1
      fi
      SAVE_EVERY_STEPS="$2"
      shift 2
      ;;
    --save_every_steps=*|--save-every-steps=*)
      SAVE_EVERY_STEPS="${1#*=}"
      shift 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ "${MODE}" != "fit" && "${MODE}" != "predict" ]]; then
  echo "[ERROR] MODE must be fit or predict, got: ${MODE}"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}"
  exit 1
fi

if ! [[ "${SAVE_EVERY_STEPS}" =~ ^[0-9]+$ ]] || [[ "${SAVE_EVERY_STEPS}" -lt 1 ]]; then
  echo "[ERROR] save_every_steps must be an integer >= 1, got: ${SAVE_EVERY_STEPS}"
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

DINO_ARGS=()
if [[ -n "${DINO_PATH}" ]]; then
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
  DINO_ARGS=(--model.diffusion_trainer.init_args.encoder.init_args.weight_path "${DINO_PATH}")
elif [[ "${MODE}" == "fit" ]] && grep -q "/path/to/dinov2_vitb14" "${CONFIG}"; then
  echo "[ERROR] DINO_PATH is empty."
  echo "        Pass arg #6 or set env DINO_WEIGHT_PATH."
  echo "        Example:"
  echo "        export DINO_WEIGHT_PATH=/path/to/facebookresearch_dinov2_main/dinov2_vitb14"
  exit 1
fi

supports_tags_exp=false
if [[ -f main.py ]] && grep -Eq "nested_key *= *[\"']tags[\"']" main.py; then
  supports_tags_exp=true
fi

has_tags_exp_override=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--tags.exp" || "${arg}" == --tags.exp=* ]]; then
    has_tags_exp_override=true
    break
  fi
done

if [[ "${supports_tags_exp}" == "false" && "${has_tags_exp_override}" == "true" ]]; then
  FILTERED_EXTRA_ARGS=()
  skip_next=false
  for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "${skip_next}" == "true" ]]; then
      skip_next=false
      continue
    fi
    case "${arg}" in
      --tags.exp)
        echo "[WARN] Current branch does not expose --tags.exp. Dropping that override."
        skip_next=true
        ;;
      --tags.exp=*)
        echo "[WARN] Current branch does not expose --tags.exp. Dropping that override."
        ;;
      *)
        FILTERED_EXTRA_ARGS+=("${arg}")
        ;;
    esac
  done
  EXTRA_ARGS=("${FILTERED_EXTRA_ARGS[@]}")
  has_tags_exp_override=false
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
TAG_EXP_SANITIZED="$(echo "${RUN_NAME}" | sed 's#[/: ]#_#g')"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Branch: ${BRANCH}"
echo "[INFO] Mode: ${MODE}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Data root: ${DATA_ROOT}"
echo "[INFO] Wandb project/run: ${PROJECT}/${RUN_NAME}"
echo "[INFO] Save checkpoint every ${SAVE_EVERY_STEPS} train steps"
if [[ -n "${DINO_PATH}" ]]; then
  echo "[INFO] DINO path: ${DINO_PATH}"
fi
if [[ "${supports_tags_exp}" == "true" ]]; then
  echo "[INFO] tags.exp: ${TAG_EXP_SANITIZED}"
fi
if [[ -n "${CKPT_DIR}" ]]; then
  echo "[INFO] Checkpoint dir: ${CKPT_DIR}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "[INFO] Extra args: ${EXTRA_ARGS[*]:-(none)}"

if [[ -n "${CKPT_DIR}" ]]; then
  mkdir -p "${CKPT_DIR}"
  export PIXNERD_CKPT_DIR="${CKPT_DIR}"
fi
export PIXNERD_SAVE_EVERY_N_TRAIN_STEPS="${SAVE_EVERY_STEPS}"

cmd=(
  python main.py "${MODE}" -c "${CONFIG}"
  --data.train_dataset.init_args.root "${DATA_ROOT}"
  --trainer.logger.init_args.project "${PROJECT}"
  --trainer.logger.init_args.name "${RUN_NAME}"
)

if [[ ${#DINO_ARGS[@]} -gt 0 ]]; then
  cmd+=("${DINO_ARGS[@]}")
fi

if [[ "${supports_tags_exp}" == "true" && "${has_tags_exp_override}" == "false" ]]; then
  cmd+=(--tags.exp "${TAG_EXP_SANITIZED}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Running: ${cmd[*]}"
"${cmd[@]}"
