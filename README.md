# CV18-ThesisProject

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

## 1. Install CAP4D

```bash
git clone https://github.com/felixtaubner/cap4d/
cd cap4d

conda create --name cap4d_env python=3.10
conda activate cap4d_env

pip install -r requirements.txt
export PYTHONPATH=$(realpath "./"):$PYTHONPATH
```

Install PyTorch3D with CUDA support (build from source is recommended):

```bash
export FORCE_CUDA=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

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

Check `examples/debug_output/tesla/sequence_00/renders.mp4` — if it shows a blurry cartoon Tesla,
the install is good.

## 4. Custom inference

To run on your own images/video instead of the provided examples, first install Pixel3DMM for 3D face
tracking (this step is separate from the main `cap4d_env`, and is prone to package version conflicts —
report issues upstream if you hit them):

```bash
export FLAME_USERNAME=your_flame_user_name
export FLAME_PWD=your_flame_password
export PIXEL3DMM_PATH=$(realpath "../path/to/pixel3dmm")   # where to clone Pixel3DMM
export CAP4D_PATH=$(realpath "./")

bash scripts/install_pixel3Dmm.sh
```

Then run the pipeline stages in order:

**a. Track reference images/video and (optionally) a driving video**

```bash
export PIXEL3DMM_PATH=$(realpath "../path/to/pixel3dmm")
export CAP4D_PATH=$(realpath "./")

mkdir -p examples/output/custom/

# Reference: a directory of frames is treated as discontinuous monocular images
bash scripts/track_video_pixel3dmm.sh examples/input/felix/images/cam0/ examples/output/custom/reference_tracking/

# Driving video: a single file is treated as a continuous monocular video
bash scripts/track_video_pixel3dmm.sh examples/input/animation/example_video.mp4 examples/output/custom/driving_video_tracking/
```

**b. Generate multi-view images with the MMDM**

```bash
python cap4d/inference/generate_images.py \
    --config_path configs/generation/default.yaml \
    --reference_data_path examples/output/custom/reference_tracking/ \
    --output_path examples/output/custom/mmdm/
```

**c. Fit the Gaussian avatar**

```bash
python gaussianavatars/train.py \
    --config_path configs/avatar/default.yaml \
    --source_paths examples/output/custom/mmdm/reference_images/ examples/output/custom/mmdm/generated_images/ \
    --model_path examples/output/custom/avatar/ \
    --interval 5000
```

> GaussianAvatars compiles CUDA extensions at install time. If `conda activate` exposes an old system
> GCC and the build fails, pin a newer compiler (e.g. GCC ≥11) and `CUDA_HOME` explicitly before running
> this step, either in the same env or a dedicated one — this repo's SLURM scripts (below) do this via a
> separate `cap4d_stable` env with GCC 13.2 and CUDA 12.8 pinned.

**d. Animate the avatar**

```bash
python gaussianavatars/animate.py \
    --model_path examples/output/custom/avatar/ \
    --target_animation_path examples/output/custom/driving_video_tracking/fit.npz \
    --target_cam_trajectory_path examples/output/custom/driving_video_tracking/cam_static.npz \
    --output_path examples/output/custom/animation_example/ \
    --export_ply 1 --compress_ply 0
```

Or run steps a–d together:

```bash
bash scripts/generate_avatar.sh {INPUT_VIDEO_PATH} {OUTPUT_PATH} [{QUALITY}] [{DRIVING_VIDEO_PATH}]
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

Dependency graph:

```
01 ──> 03.a ──> 03 ──> 04 ──┐
                             ├──> 05
02 ───────────────────────── ┘
```

Update `CAP4D_PATH`, `PIXEL3DMM_PATH`, and input paths near the top of each script to match your setup,
then submit the whole chain with correct SLURM `--dependency` ordering:

```bash
chmod +x submit_pipeline.sh
./submit_pipeline.sh
```

Or submit an individual step, e.g. to rerun just avatar fitting:

```bash
sbatch 04_cap4d_fit_avatar.sbatch
```

Track jobs with `squeue -u $USER`.

**Why separate SLURM jobs instead of one script:** steps 1–3 and steps 4–5 use different conda
environments and compiler pins; step 3 and step 4 can each run for hours, so a single combined job would
waste GPU time already spent on earlier successful steps if a later step fails; and separate jobs let you
resubmit just the failed stage instead of the whole pipeline.

**Note on `03.a`:** video-mode reference tracking saves color frames as `images/cam0.mp4`, but
`ReferenceDataset` expects an extensionless path with the same name (matching how directories of frames
are stored). This script creates that symlink automatically.

## Output layout

```
examples/output/<sample_name>/
├── reference_tracking/   # Step 1 output (+ step 3a symlink fix)
├── driving_tracking/     # Step 2 output
├── mmdm/                 # Step 3 output (reference_images/, generated_images/)
├── avatar/               # Step 4 output (fitted Gaussian avatar)
└── animation/             # Step 5 output, incl. exported_animation.ply
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
