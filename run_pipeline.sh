#!/bin/bash
# run_pipeline.sh
#
# Runs the full CAP4D custom-inference pipeline (tracking -> MMDM generation ->
# avatar fitting -> animation) on a single non-SLURM machine.
#
# Unlike scripts/generate_avatar.sh (upstream), this script does NOT assume one
# conda environment can run the whole pipeline. Gaussian avatar fitting needs a
# newer compiler/CUDA than tracking + MMDM generation to build GaussianAvatars'
# CUDA extensions (see README > Requirements), so this script activates
# cap4d_env for stage 1, then switches to cap4d_stable for stage 2.
#
# Usage: edit the "Set your paths" block below, then run:
#   bash run_pipeline.sh

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
# Only edit these — same variables used throughout the README and every
# .sbatch script in this repo.
CAP4D_PATH=/path/to/cap4d
PIXEL3DMM_PATH=/path/to/pixel3dmm

REFERENCE_INPUT=/path/to/your/reference_images_or_video   # directory of images, OR a single video file
DRIVING_INPUT=/path/to/your/driving_video.mp4              # optional — leave empty ("") to skip animation

OUTPUT_ROOT=${CAP4D_PATH}/examples/output/custom
# ──────────────────────────────────────────────────────────────────────────

# ── System-specific settings — adjust for your machine ─────────────────────
CONDA_SH_PATH=/path/to/anaconda/etc/profile.d/conda.sh
GCC_BIN=/path/to/gnu-compilers/bin            # needs GCC >= 11
CUDA_HOME_PATH=/path/to/cuda                  # e.g. /usr/local/cuda-12.8
# ──────────────────────────────────────────────────────────────────────────

source "${CONDA_SH_PATH}"

REFERENCE_TRACKING=${OUTPUT_ROOT}/reference_tracking
DRIVING_TRACKING=${OUTPUT_ROOT}/driving_tracking
MMDM_OUTPUT=${OUTPUT_ROOT}/mmdm
AVATAR_OUTPUT=${OUTPUT_ROOT}/avatar
ANIMATION_OUTPUT=${OUTPUT_ROOT}/animation

mkdir -p "${OUTPUT_ROOT}"
cd "${CAP4D_PATH}"

# ── Stage 1: tracking + MMDM generation (cap4d_env) ─────────────────────────
conda activate cap4d_env
export PIXEL3DMM_PATH CAP4D_PATH
export PYTHONPATH="${CAP4D_PATH}:${PIXEL3DMM_PATH}:${PYTHONPATH:-}"

echo "=== [1/4] Tracking reference input ==="
bash scripts/track_video_pixel3dmm.sh "${REFERENCE_INPUT}" "${REFERENCE_TRACKING}"

if [ -n "${DRIVING_INPUT}" ]; then
    echo "=== [1/4] Tracking driving input ==="
    bash scripts/track_video_pixel3dmm.sh "${DRIVING_INPUT}" "${DRIVING_TRACKING}"
fi

# Fix extensionless symlink if REFERENCE_INPUT was a video file rather than a
# directory of frames — see README for why this is needed.
IMAGES_DIR=${REFERENCE_TRACKING}/images
for video_file in "${IMAGES_DIR}"/*.mp4; do
    [ -e "${video_file}" ] || continue
    cam_name=$(basename "${video_file}" .mp4)
    link_path="${IMAGES_DIR}/${cam_name}"
    [ -e "${link_path}" ] && continue
    ln -s "$(basename "${video_file}")" "${link_path}"
    echo "Created symlink: ${link_path} -> $(basename "${video_file}")"
done

echo "=== [2/4] Generating multi-view images with MMDM ==="
echo "(this can take several hours)"
python cap4d/inference/generate_images.py \
    --config_path configs/generation/default.yaml \
    --reference_data_path "${REFERENCE_TRACKING}" \
    --output_path "${MMDM_OUTPUT}"

conda deactivate

# ── Stage 2: avatar fitting + animation (cap4d_stable) ──────────────────────
conda activate cap4d_stable
export CC="${GCC_BIN}/gcc"
export CXX="${GCC_BIN}/g++"
export CUDA_HOME="${CUDA_HOME_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${CAP4D_PATH}:${PYTHONPATH:-}"

echo "=== [3/4] Fitting Gaussian avatar ==="
python gaussianavatars/train.py \
    --config_path configs/avatar/default.yaml \
    --source_paths "${MMDM_OUTPUT}/reference_images/" "${MMDM_OUTPUT}/generated_images/" \
    --model_path "${AVATAR_OUTPUT}" \
    --interval 5000

if [ -n "${DRIVING_INPUT}" ]; then
    echo "=== [4/4] Animating avatar ==="
    python gaussianavatars/animate.py \
        --model_path "${AVATAR_OUTPUT}" \
        --target_animation_path "${DRIVING_TRACKING}/fit.npz" \
        --target_cam_trajectory_path "${DRIVING_TRACKING}/cam_static.npz" \
        --output_path "${ANIMATION_OUTPUT}" \
        --export_ply 1 --compress_ply 0
    echo "=== Done ==="
    echo "Output animation: ${ANIMATION_OUTPUT}/exported_animation.ply"
    echo "Upload this file to: https://felixtaubner.github.io/cap4d/viewer/"
else
    echo "=== Done ==="
    echo "No DRIVING_INPUT set — skipping animation. Fitted avatar is at: ${AVATAR_OUTPUT}"
fi
