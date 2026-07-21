# VGGS: VGGT-guided Gaussian Splatting for Efficient and Faithful Sparse-View Surface Reconstruction
[Peng Xiang](https://scholar.google.com/citations?user=Bp-ceOAAAAAJ&hl=zh-CN&oi=sra), Liang Han, [Hui Zhang](https://www.thss.tsinghua.edu.cn/en/faculty/huizhang.htm), [Yu-Shen Liu](http://cgcad.thss.tsinghua.edu.cn/liuyushen/), [Zhizhong Han](https://h312h.github.io/)

This working copy is based on the upstream VGGS repository provided by the user:
[AllenXiangX/VGGS](https://github.com/AllenXiangX/VGGS). It has been adapted
for a Python 3.12 + uv workflow, with additional Jetson AGX Orin notes and a few
training-stability changes intended to improve sparse-view reconstruction.



[<img src="./assets/teaser.png" width="100%" alt="Intro pic" />](assets/teaser.png)

## [VGGS]

> Reconstructing a faithful geometric surface from sparse images remains a fundamental challenge in 3D computer vision. While recent methods have achieved remarkable progress, they still struggle to recover reliable geometry due to the lack of multi-view geometric cues, particularly in non-overlapping regions. To address this issue, we introduce VGGS, a Gaussian Splatting (GS) method that exploits multi-view geometric priors from VGGT for efficient and high-fidelity sparse-view surface reconstruction. Our primary contribution is an anchor-calibrated depth estimation scheme, which yields accurate depth maps. The insight is to align the VGGT depth prior to the underlying surface with a sparse set of multi-view consistent anchors, then infer depth for unreliable regions by relative depth estimation. Furthermore, to mitigate misalignment in complex scenes, we propose a relative depth consistency loss that penalizes the rendered depth if its relative depth relationship in local regions is inconsistent to the multi-view prior. Extensive experiments on widely-used benchmarks show that VGGS surpasses state-of-the-art methods in both accuracy and efficiency, delivering 4–7× faster optimization while reducing memory consumption compared to previous GS-based approaches.

## Changes in this working copy

- Replaced the original conda/Python 3.8 setup with Python 3.12, `uv`,
  `pyproject.toml`, and `uv.lock`.
- Removed legacy `requirements*.txt` files; runtime dependencies now live in
  `pyproject.toml`.
- Kept PyTorch, TorchVision, TorchAudio, PyTorch3D, and the local CUDA
  extensions as explicit install steps so they can match the target CUDA or
  JetPack stack.
- Added Jetson AGX Orin / JetPack guidance in
  [docs/python312-jetson.md](docs/python312-jetson.md).
- Added adaptive VGGT depth-confidence masking via `depth_conf_keep_ratio` and
  `depth_conf_keep_ratio_2`.
- Replaced the plain depth scale/shift solve with a confidence-weighted robust
  alignment controlled by `robust_depth_align`, `depth_align_irls_iters`, and
  `depth_align_min_anchors`.
- Added ramp-up scheduling for geometry-related losses through
  `loss_ramp_iters`, reducing early over-constraint from VGGT depth and normal
  priors.
- Added edge-aware filtering for depth supervision through
  `depth_edge_keep_ratio`, which avoids applying VGGT depth losses on likely
  image discontinuities.
- Added EMA smoothing for generated pseudo-depth maps through
  `pseudo_depth_ema`, reducing jitter between pseudo-label refreshes.
- Added `scripts/sweep_quality.py` to run small ablation sweeps over the new
  confidence, edge-mask, EMA, and loss-ramp parameters.
- Added `train_optuna.py` for automated hyperparameter tuning of the
  `train.py` quality knobs. `scripts/optuna_quality.py` remains as a
  compatibility wrapper.
- Added optional TSDF edge filtering in `render.py` and `scripts/render_tnt.py`
  through `--tsdf_edge_keep_ratio`.

## Installation

This fork supports the original desktop/Linux workflow and a Jetson AGX Orin
workflow. The Jetson path is more constrained because PyTorch, CUDA extensions,
PyTorch3D, and Open3D must match JetPack/aarch64.

### Standard Linux / x86_64

```shell
# SSH
git clone https://github.com/AllenXiangX/VGGS.git
cd VGGS

uv venv --python 3.12
source .venv/bin/activate
uv sync --extra tuning --extra eval --extra metrics

# Install a PyTorch stack matching your CUDA driver. Example for CUDA 11.8:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch3D and local CUDA extensions.
uv pip install pytorch3d
export TORCH_CUDA_ARCH_LIST="native"
uv pip install -v --no-build-isolation submodules/diff-plane-rasterization
uv pip install -v --no-build-isolation submodules/simple-knn
```

### Jetson AGX Orin / JetPack

Use NVIDIA Jetson PyTorch wheels or an NVIDIA Jetson container. Do not install
PyTorch from `https://download.pytorch.org/whl/cu*` on Jetson: those wheels are
for desktop/server Linux, not aarch64 Jetson.

Start from the repo root:

```shell
cd ~/Desktop/VGGS
uv venv --python 3.12
source .venv/bin/activate

# Keep Open3D and PyPI lpips out of the base Jetson environment.
# open3d has no PyPI aarch64 wheel, and lpips can pull a generic PyTorch stack.
uv sync --extra tuning
```

Install the NVIDIA-provided PyTorch stack for your JetPack/CUDA version. Example
shape, replacing the URL with the exact wheel for your JetPack from NVIDIA:

```shell
uv pip uninstall torch torchvision torchaudio triton
export TORCH_INSTALL="https://developer.download.nvidia.com/compute/redist/jp/.../torch-...-linux_aarch64.whl"
uv pip install --no-cache-dir "$TORCH_INSTALL"
```

Verify before building anything:

```shell
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

For Jetson AGX Orin, the GPU compute capability is 8.7. Export the build
environment before compiling extensions:

```shell
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=2
```

Install PyTorch3D. On Jetson/aarch64 this often needs a source build:

```shell
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```

Build the local CUDA extensions against the already-installed Jetson PyTorch:

```shell
uv pip install -v --no-build-isolation submodules/diff-plane-rasterization
uv pip install -v --no-build-isolation submodules/simple-knn
```

Important details for Jetson:

- Always use `--no-build-isolation` for the CUDA submodules. Otherwise the build
  environment cannot see your installed Jetson PyTorch and fails with
  `ModuleNotFoundError: No module named 'torch'`.
- `simple-knn` needs `#include <cfloat>` in `submodules/simple-knn/simple_knn.cu`
  for CUDA 12.6/Python 3.12 builds, otherwise `FLT_MAX` is undefined. This fork
  includes that patch.
- `open3d` is optional in `pyproject.toml` under the `eval` extra and is only
  installed automatically on x86_64. It is needed for `render.py` and DTU/TNT
  mesh evaluation. On Jetson, either build Open3D manually, use a compatible
  container, or run rendering/evaluation on an x86_64 machine.
- After manually installing Jetson PyTorch, avoid plain `uv sync` if it would
  remove manual packages. Use `uv sync --extra tuning --inexact` when needed.
- If Torch warns that the installed build does not support Orin CC 8.7, replace
  it with a JetPack-compatible NVIDIA build before training.

#### Open3D on Jetson

Open3D is only needed for TSDF mesh extraction and DTU/TNT mesh evaluation. The
Gaussian Splatting checkpoint itself can still be rendered to images without
Open3D by passing `--skip_mesh` to `render.py`.

First try the prebuilt wheel in the active VGGS environment:

```shell
cd ~/Desktop/VGGS
source .venv/bin/activate
python -m pip install -U pip
python -m pip install open3d
python - <<'PY'
import open3d as o3d
print(o3d.__version__)
PY
```

If no compatible aarch64 wheel is available, build a minimal Open3D from source.
This build disables GUI, WebRTC, CUDA, ML ops, tests, and examples because VGGS
only needs Open3D's CPU TSDF integration and mesh I/O on Jetson.

```shell
sudo apt update
sudo apt install -y \
  git build-essential pkg-config ccache \
  libgl1-mesa-dev libglu1-mesa-dev \
  libx11-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  libjpeg-dev libpng-dev libtiff-dev \
  libeigen3-dev

cd ~/Desktop/VGGS
source .venv/bin/activate
python -m pip install -U pip wheel setuptools cmake ninja
```

Open3D can require a lot of RAM while compiling. Add temporary swap if the build
is killed or the Jetson becomes unstable:

```shell
sudo fallocate -l 8G /swapfile-open3d
sudo chmod 600 /swapfile-open3d
sudo mkswap /swapfile-open3d
sudo swapon /swapfile-open3d
```

Clone and configure Open3D:

```shell
cd ~
git clone --recursive https://github.com/isl-org/Open3D.git
cd Open3D
git checkout v0.19.0
git submodule update --init --recursive
./util/install_deps_ubuntu.sh

mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_MODULE=ON \
  -DBUILD_CUDA_MODULE=OFF \
  -DBUILD_GUI=OFF \
  -DBUILD_WEBRTC=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_PYTORCH_OPS=OFF \
  -DBUILD_TENSORFLOW_OPS=OFF \
  -DPython3_EXECUTABLE="$HOME/Desktop/VGGS/.venv/bin/python"
```

Build with a low job count on AGX Orin:

```shell
make -j2
make install-pip-package -j2
```

Verify the Python module from the VGGS environment:

```shell
cd ~/Desktop/VGGS
source .venv/bin/activate
python - <<'PY'
import open3d as o3d
print("Open3D:", o3d.__version__)
mesh = o3d.geometry.TriangleMesh.create_box()
print(mesh)
PY
```

You can now extract a mesh with `render.py` by omitting `--skip_mesh`:

```shell
python render.py \
  -m exp/optuna/dtu_scan24/optuna_trial_0020 \
  --iteration 3000 \
  --config configs/dtu.yaml \
  --skip_test
```

Expected mesh outputs:

```shell
exp/optuna/dtu_scan24/optuna_trial_0020/mesh/tsdf_fusion.ply
exp/optuna/dtu_scan24/optuna_trial_0020/mesh/tsdf_fusion_post.ply
```

If Open3D is not installed yet, render images/depth/normals only:

```shell
python render.py \
  -m exp/optuna/dtu_scan24/optuna_trial_0020 \
  --iteration 3000 \
  --config configs/dtu.yaml \
  --skip_test \
  --skip_mesh
```

Quick import checks:

```shell
python - <<'PY'
import torch
import simple_knn._C
import diff_plane_rasterization
import pytorch3d
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("VGGS CUDA deps OK")
PY
```

For more Jetson notes, see [docs/python312-jetson.md](docs/python312-jetson.md).

## Dataset
Please download the preprocessed [DTU dataset and TNT dataset](https://drive.google.com/drive/folders/1x62cuv46E-elH-zeIQrj9NFMlpw1Vf04?usp=drive_link). The DTU ground truth (dtu_eval) can be downloaded from [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36)


The data folder should like this:
```shell
data
├── DTU
│   ├── set_22_25_28
│   │   ├── scan24
│   │   │   ├── images
│   │   │   ├── mask
│   │   │   ├── dense
│   │   │   │   │── sparse
│   │   │   │   │── depth_vggt
│   │   │   │   │── images
│   │   │   │   │── normal
│   │   │   └── cameras.npz
│   │   └── ...
│   ├── dtu_eval
│   │   ├── Points
│   │   │   └── stl
│   │   └── ObsMask
├── tnt_dataset
│   ├── tnt_10views
│   │   ├── Ignatius
│   │   │   ├── images
│   │   │   ├── depth_vggt
│   │   │   ├── sparse
│   │   │   ├── normal
│   │   │   ├── cameras.json
│   │   │   ├── Ignatius_COLMAP_SfM.log
│   │   │   ├── Ignatius_trans.txt
│   │   │   ├── Ignatius.json
│   │   │   └── Ignatius.ply
│   │   └── ...
```
## Training and Evaluation
```shell
# Fill in the relevant parameters in the script, then run it.

# DTU dataset
python scripts/run_dtu.py

# Tanks and Temples dataset
python scripts/run_tnt.py
```

### Quality sweep

The added sweep helper prints commands by default, so runs can be inspected
before launching them:

```shell
python scripts/sweep_quality.py \
  -s data/DTU/set_22_25_28/scan24/dense \
  -m exp/sweeps/dtu_scan24 \
  --config configs/dtu.yaml
```

Add `--run` to execute the generated commands.

### Optuna tuning

Optuna is useful for this fork because the best values for confidence masking,
edge filtering, EMA smoothing, and geometry-loss weights are scene- and dataset-
dependent. The script optimizes the PSNR printed by `train.py`; for geometry
metrics such as Chamfer distance, extend the objective to call the mesh
evaluation after each trial.

```shell
uv sync --extra tuning

python train_optuna.py \
  -s data/DTU/set_22_25_28/scan24/dense \
  -m exp/optuna/dtu_scan24 \
  --config configs/dtu.yaml \
  --n-trials 20 \
  --sampler tpe \
  --pruner none \
  --storage sqlite:///exp/optuna/dtu_scan24.db \
  --report-dir exp/optuna/dtu_scan24_report
```

`train.py` itself remains the deterministic training entry point. `train_optuna.py`
imports Optuna directly, creates the study, repeatedly launches `train.py`,
tunes training knobs, and records `tsdf_edge_keep_ratio` as a trial attribute
for render/export experiments.

At the end of the study, `train_optuna.py` writes a benchmark report containing:

- `study_report.html`: a self-contained visual report with summary cards, best
  trials, a score timeline, and every trial's parameters/performance.
- `trials.csv`: flat table for spreadsheet analysis.
- `trials.json`: machine-readable study export.

### Tanks and Temples Optuna tuning

The preprocessed Tanks and Temples 10-view scenes are expected under:

```shell
data/tnt_dataset/tnt_10views/
├── Barn
├── Caterpillar
├── Ignatius
└── Truck
```

Each scene must contain at least:

```shell
images/
depth_vggt/
normal/
sparse/
<Scene>_COLMAP_SfM.log
<Scene>_trans.txt
<Scene>.json
<Scene>.ply
```

`depth_vggt` and `normal` are VGGS/VGGT-specific priors; the raw official Tanks
and Temples download does not provide them. Prefer the preprocessed VGGS dataset
when available. If starting from the official Tanks and Temples site, generate
COLMAP data first with `scripts/preprocess/convert_tnt.py`, then generate the
VGGT depth/confidence and normal `.npy` files before training.

Run Optuna on one TNT scene:

```shell
python train_optuna.py \
  -s data/tnt_dataset/tnt_10views/Ignatius \
  -m exp/optuna/tnt_10views/Ignatius \
  --config configs/tnt_10views.yaml \
  --n-trials 20 \
  --test-iteration 3000 \
  --metric-split train \
  --sampler tpe \
  --pruner none \
  --storage sqlite:///exp/optuna/tnt_10views/Ignatius/ignatius_optuna.db \
  --study-name vggs-tnt-ignatius \
  --report-dir exp/optuna/tnt_10views/Ignatius/report \
  --common-args "-r2 --ncc_scale 0.5 --data_device cuda --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation"
```

Run the helper on all preprocessed TNT 10-view scenes:

```shell
python scripts/run_tnt_optuna.py \
  --data-root data/tnt_dataset/tnt_10views \
  --output-root exp/optuna/tnt_10views \
  --n-trials 20
```

Use `--dry-run` to print the commands without launching training:

```shell
python scripts/run_tnt_optuna.py --scenes Ignatius --n-trials 1 --dry-run
```

TNT scenes in this repo do not include a `split.json` by default, so use
`--metric-split train` unless you add your own train/test split file. Also avoid
passing `--quiet` through `--common-args`, because Optuna parses the PSNR from
`train.py` stdout.

### Official Tanks and Temples Submission

The official benchmark requires a complete `intermediate`, `advanced`, or `both`
submission set. A single scene is not accepted. The upload folder must contain
one `<Scene>.ply` point cloud and one `<Scene>.log` camera trajectory file per
scene, plus the official uploader and your credentials file.

The helper below prepares the official Intermediate workflow. It downloads the
official image sets through the Tanks and Temples downloader, runs COLMAP, writes
TNT `.log` files, launches VGGS/Optuna once VGGT priors are available, and
assembles the upload folder from the best trials.

```shell
python scripts/tnt_official_pipeline.py   --group intermediate   --download   --prepare-colmap
```

Before training, each scene must contain the VGGS priors:

```shell
data/tnt_official/intermediate/<Scene>/depth_vggt/*_pred.npy
data/tnt_official/intermediate/<Scene>/depth_vggt/*_conf.npy
data/tnt_official/intermediate/<Scene>/normal/*_normal.npy
```

Those files are not provided by the official Tanks and Temples image sets and
must be generated by a VGGT/depth-normal preprocessing step before VGGS training.
Once they exist, run:

```shell
python scripts/tnt_official_pipeline.py   --group intermediate   --train   --n-trials 20
```

After all Intermediate scenes have completed, assemble the upload folder:

```shell
python scripts/prepare_tnt_submission.py   --group intermediate   --data-root data/tnt_official   --output-root exp/tnt_official   --submission-dir submission_tnt/intermediate
```

Copy the official uploader and your Tanks and Temples credentials file into the
submission directory, then upload from that directory:

```shell
cp tools/tanksandtemples/upload_t2_results.py submission_tnt/intermediate/
cd submission_tnt/intermediate
python upload_t2_results.py --group intermediate
```

Use `--dry-run` to inspect the full pipeline without downloading, training, or
writing outputs:

```shell
python scripts/tnt_official_pipeline.py   --group intermediate   --download   --prepare-colmap   --train   --dry-run
```

## [Cite this work]

```
@inproceedings{xiang2026vggs,
  title={VGGS: VGGT-guided Gaussian Splatting for Efficient and Faithful Sparse-View Surface Reconstruction},
  author={Xiang, Peng and Han, Liang and Zhang, Hui and Liu, Yu-Shen and Han, Zhizhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={13},
  pages={10969--10977},
  year={2026}
}

```

## Acknowledgements

This work is built upon: 
- [PGSR](https://github.com/zju3dv/PGSR)


We thank the authors for their great job!
