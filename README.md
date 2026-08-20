# CAP4D Avatar Pipeline

This repo runs the [CAP4D](https://felixtaubner.github.io/cap4d/) pipeline, turning one or more reference
images (or a short reference video) into an animatable 4D portrait avatar, driven by a separate video.

> CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models
> Felix Taubner, Ruihang Zhang, Mathieu Tuli, David B. Lindell — CVPR 2025 (Oral)
> [Project page](https://felixtaubner.github.io/cap4d/) · [Paper repo](https://github.com/felixtaubner/cap4d)

CAP4D itself has no dependency on SLURM, it just needs to run on a machine (workstation, cloud VM, or
a SLURM-managed cluster) that meets the GPU/CPU/RAM requirements below. This repo includes SLURM batch
scripts for convenience when running on a shared HPC cluster, but the same commands run fine standalone.

## Requirements

- NVIDIA GPU with CUDA support. Multi-view generation (step 2 below) uses all visible CUDA devices
  automatically; more GPUs = faster generation.
- **≥64 GB RAM recommended** for the MMDM image generation step; it can run for several hours.
- Python 3.10 + conda.
- A [FLAME](https://flame.is.tue.mpg.de/) account (free, for downloading FLAME blendshapes).

This pipeline needs **two separate conda environments**, because tracking/generation and avatar
fitting/animation were validated against different CUDA/PyTorch builds and don't install cleanly
together:

| Env | Used for | CUDA / PyTorch | Requirements file |
|-----|----------|-----------------|--------------------|
| `cap4d_env` | Steps 1–3: Pixel3DMM tracking, MMDM generation | CUDA 11.8, torch 2.3.1 | [`requirements-tracking.txt`](requirements-tracking.txt) |
| `cap4d_stable` | Steps 4–5: Gaussian avatar fitting, animation | CUDA 12.4+, torch 2.6.0 | [`requirements-avatar.txt`](requirements-avatar.txt) |

Each file has PyTorch install instructions at the top; install the CUDA-matched PyTorch build
first, then `pip install -r <file>`, since plain `pip install -r requirements.txt` won't reliably
resolve the correct CUDA-tagged wheel on its own. See each file's header comment for the exact
commands and for the system-level prerequisites (CUDA toolkit version, compiler) that pip can't
install.

## 1. Install CAP4D (tracking + generation environment)

```bash
git clone https://github.com/felixtaubner/cap4d/
cd cap4d

conda create --name cap4d_env python=3.10
conda activate cap4d_env

# Install the CUDA-matched PyTorch build first (see requirements-tracking.txt header)
pip install torch==2.3.1+cu118 torchvision==0.18.1 torchaudio==2.3.1 \
    --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements-tracking.txt
export PYTHONPATH=$(realpath "./"):$PYTHONPATH
```

Install PyTorch3D with CUDA support (build from source is recommended, so it's compiled against the
torch build above):

```bash
export FORCE_CUDA=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Avatar fitting and animation (steps 4–5) use a **second environment**, `cap4d_stable`, set that up
separately using [`requirements-avatar.txt`](requirements-avatar.txt) when you get there (see step 4c
below).

## 2. Download FLAME and MMDM weights

```bash
export FLAME_USERNAME=your_flame_user_name
export FLAME_PWD=your_flame_password

bash scripts/download_flame.sh
bash scripts/download_mmdm_weights.sh
```

If the FLAME download script fails, manually download FLAME2023 from the FLAME website, place
`flame2023_no_jaw.pkl` in `data/assets/flame/`, then fix it for newer numpy:

```bash
python scripts/fixes/fix_flame_pickle.py --pickle_path data/assets/flame/flame2023_no_jaw.pkl
```

## 3. Verify the install

```bash
bash scripts/test_pipeline.sh
```

Check `examples/debug_output/tesla/sequence_00/renders.mp4`, if it shows a blurry cartoon Tesla,
the install is good.

## 4. Custom inference

To run on your own images/video instead of the provided examples, first install Pixel3DMM for 3D face
tracking (this step is separate from the main `cap4d_env`, and is prone to package version conflicts,
report issues upstream if you hit them):

```bash
export FLAME_USERNAME=your_flame_user_name
export FLAME_PWD=your_flame_password
export PIXEL3DMM_PATH=$(realpath "../path/to/pixel3dmm")   # where to clone Pixel3DMM
export CAP4D_PATH=$(realpath "./")

bash scripts/install_pixel3Dmm.sh
```

### Set your paths

This is the **only place you need to edit paths**. Every command below reuses these variables as-is.

```bash
# ── Paths ──────────────────────────────────────────────────────────────────
CAP4D_PATH=$(realpath "./")
PIXEL3DMM_PATH=$(realpath "../path/to/pixel3dmm")

REFERENCE_INPUT=/path/to/your/reference_images_or_video   # directory of images, OR a single video file
DRIVING_INPUT=/path/to/your/driving_video.mp4              # optional — only needed if animating with a driving video

OUTPUT_ROOT=${CAP4D_PATH}/examples/output/custom
# ──────────────────────────────────────────────────────────────────────────

export CAP4D_PATH PIXEL3DMM_PATH
REFERENCE_TRACKING=${OUTPUT_ROOT}/reference_tracking
DRIVING_TRACKING=${OUTPUT_ROOT}/driving_tracking
MMDM_OUTPUT=${OUTPUT_ROOT}/mmdm
AVATAR_OUTPUT=${OUTPUT_ROOT}/avatar
ANIMATION_OUTPUT=${OUTPUT_ROOT}/animation

mkdir -p "${OUTPUT_ROOT}"
```

Option A: run the whole pipeline in one command (recommended)

run_pipeline.sh runs tracking, MMDM generation, avatar fitting, and animation back to back, including switching from cap4d_env to cap4d_stable partway through, and applying the reference symlink fix automatically if needed:

```bash
chmod +x run_pipeline.sh
bash run_pipeline.sh
```

Edit the path block at the top of run_pipeline.sh first (same variables as above, plus a few system-specific settings for your compiler/CUDA install. See the comments in the file).

> Why not scripts/generate_avatar.sh? Upstream's own "run everything" script assumes
> a single conda environment can handle the whole pipeline. On this setup, that's not true:
> Gaussian avatar fitting needs a newer compiler/CUDA to build GaussianAvatars' CUDA extensions
> than tracking/MMDM generation does (see Requirements), so generate_avatar.sh will very likely
> fail partway through, after already spending the hours-long MMDM generation step.
> run_pipeline.sh does the same job but switches environments at the right point.

Use Option B instead if you want to inspect/rerun individual stages, or if your `REFERENCE_INPUT` is a
video file (see the symlink note in step a below, which `generate_avatar.sh` does not apply automatically).

### Option B: run each stage individually

**a. Track reference images/video and (optionally) a driving video**

```bash
# Reference: a directory of frames is treated as discontinuous monocular images.
# If REFERENCE_INPUT is a single file instead, it's treated as a continuous monocular video.
bash scripts/track_video_pixel3dmm.sh "${REFERENCE_INPUT}" "${REFERENCE_TRACKING}"

# Driving video (optional, only needed for step d)
bash scripts/track_video_pixel3dmm.sh "${DRIVING_INPUT}" "${DRIVING_TRACKING}"
```

> **If `REFERENCE_INPUT` is a video file** (rather than a directory of frames), apply this fix before
> moving on to step b. `ReferenceDataset` expects `images/<camera_name>` with **no** file extension,
> matching how a directory of frames is stored — but video-mode tracking saves the color frames as
> `images/cam0.mp4` instead. Create an extensionless symlink pointing at it (decord's `VideoReader`
> sniffs file content rather than the extension, so this is safe):
>
> ```bash
> IMAGES_DIR=${REFERENCE_TRACKING}/images
>
> for video_file in "${IMAGES_DIR}"/*.mp4; do
>     [ -e "${video_file}" ] || continue
>     cam_name=$(basename "${video_file}" .mp4)
>     link_path="${IMAGES_DIR}/${cam_name}"
>     [ -e "${link_path}" ] && continue
>     ln -s "$(basename "${video_file}")" "${link_path}"
> done
> ```
>
> (SLURM users: this is exactly what `03.a_fix_reference_images_symlink.sbatch` does, see the SLURM
> section below.)

**b. Generate multi-view images with the MMDM**

```bash
python cap4d/inference/generate_images.py \
    --config_path configs/generation/default.yaml \
    --reference_data_path "${REFERENCE_TRACKING}" \
    --output_path "${MMDM_OUTPUT}"
```

**c. Fit the Gaussian avatar**

Switch to the second environment for this step and the next. GaussianAvatars compiles CUDA
extensions at install time, and needs a newer compiler/CUDA than `cap4d_env` uses. If `conda activate`
exposes an old system-default GCC, the build will fail; pin a newer compiler (e.g. GCC ≥11) and
`CUDA_HOME` explicitly:

```bash
conda create --name cap4d_stable python=3.10
conda activate cap4d_stable

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

export CC=/path/to/gcc-13/bin/gcc      # adjust to your compiler install
export CXX=/path/to/gcc-13/bin/g++
export CUDA_HOME=/usr/local/cuda-12.8  # adjust to your CUDA install
export PATH="${CUDA_HOME}/bin:${PATH}"

pip install -r requirements-avatar.txt
export PYTHONPATH=$(realpath "./"):$PYTHONPATH
```

```bash
python gaussianavatars/train.py \
    --config_path configs/avatar/default.yaml \
    --source_paths "${MMDM_OUTPUT}/reference_images/" "${MMDM_OUTPUT}/generated_images/" \
    --model_path "${AVATAR_OUTPUT}" \
    --interval 5000
```

This repo's SLURM scripts (below) do the same compiler/CUDA pinning via a dedicated `cap4d_stable`
env with GCC 13.2 and CUDA 12.8 loaded as cluster modules.

**d. Animate the avatar**

```bash
python gaussianavatars/animate.py \
    --model_path "${AVATAR_OUTPUT}" \
    --target_animation_path "${DRIVING_TRACKING}/fit.npz" \
    --target_cam_trajectory_path "${DRIVING_TRACKING}/cam_static.npz" \
    --output_path "${ANIMATION_OUTPUT}" \
    --export_ply 1 --compress_ply 0
```

## 5. View the avatar

Open the [real-time viewer](https://felixtaubner.github.io/cap4d/viewer/) (powered by Brush), click
**Load file**, and upload `exported_animation.ply` from your output directory.

## Running on a SLURM cluster (optional)

If you're running on a SLURM-managed HPC cluster rather than a standalone machine, this repo includes
batch scripts that wrap the same commands above:

| Step | Script | Env | Notes |
|------|--------|-----|-------|
| 1 | `01_cap4d_track_reference.sbatch` | `cap4d_env` | Pixel3DMM tracking on the reference image/video. |
| 2 | `02_cap4d_track_driving.sbatch` | `cap4d_env` | Pixel3DMM tracking on the driving video. Independent of step 1. |
| 3a | `03.a_fix_reference_images_symlink.sbatch` | `cap4d_env` | Fixes reference image naming for video-mode tracking (see note below). |
| 3 | `03_cap4d_generate_mmdm.sbatch` | `cap4d_env` | MMDM multi-view generation. Can take hours; wants >64 GB RAM. |
| 4 | `04_cap4d_fit_avatar.sbatch` | `cap4d_stable` | Gaussian avatar fitting; pins GCC 13.2 / CUDA 12.8 to build GaussianAvatars' CUDA extensions. |
| 5 | `05_cap4d_animate.sbatch` | `cap4d_stable` | Animates the fitted avatar with the driving tracking data. |

See [Requirements](#requirements) above for which requirements file (`requirements-tracking.txt` /
`requirements-avatar.txt`) each env needs.

Dependency graph:

```
01 ──> 03.a ──> 03 ──> 04 ───┐
                             ├──> 05
02 ──────────────────────────┘
```

Update `CAP4D_PATH`, `PIXEL3DMM_PATH`, and input paths near the top of each `.sbatch` file to match your
setup (same idea as the "Set your paths" block above; those are the only lines you should need to edit).

### Run all five steps together (recommended)

`submit_pipeline.sh` submits every step with the correct SLURM `--dependency` ordering in one go, so you
don't have to submit or track each job by hand:

```bash
chmod +x submit_pipeline.sh
./submit_pipeline.sh
```

Track progress with `squeue -u $USER`.

### Or submit an individual step

Useful for rerunning just one stage, e.g. after fixing a config and re-fitting the avatar:

```bash
sbatch 04_cap4d_fit_avatar.sbatch
```

**Why separate SLURM jobs instead of one script:** steps 1–3 and steps 4–5 use different conda
environments and compiler pins; step 3 and step 4 can each run for hours, so a single combined job would
waste GPU time already spent on earlier successful steps if a later step fails; and separate jobs let you
resubmit just the failed stage instead of the whole pipeline.

**Note on `03.a`:** only needed if your reference input is a video file rather than a directory of
frames. Video-mode reference tracking saves color frames as `images/cam0.mp4`, but `ReferenceDataset`
expects an extensionless path with the same name (matching how directories of frames are stored). This
script creates that symlink automatically. See the callout in step 4a above for what it does and why.

## Output layout

```
examples/output/<sample_name>/
├── reference_tracking/   # Step 1 output (+ step 3a symlink fix)
├── driving_tracking/     # Step 2 output
├── mmdm/                 # Step 3 output (reference_images/, generated_images/)
├── avatar/               # Step 4 output (fitted Gaussian avatar)
└── animation/            # Step 5 output, incl. exported_animation.ply
```

## Related resources

The MMDM code is based on [ControlNet](https://github.com/lllyasviel/ControlNet). The 4D Gaussian avatar
code is based on [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars). 3D face tracking uses
[Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm).

Follow-up / related work: [MVP4D](https://arxiv.org/abs/2505.24000), [CAT3D](https://cat3d.github.io/),
[FlowFace](https://arxiv.org/abs/2404.09819), [Stable Diffusion](https://arxiv.org/abs/2112.10752).

## Citation

```bibtex
@inproceedings{taubner2025cap4d,
    author    = {Taubner, Felix and Zhang, Ruihang and Tuli, Mathieu and Lindell, David B.},
    title     = {{CAP4D}: Creating Animatable {4D} Portrait Avatars with Morphable Multi-View Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5318-5330}
}
```
