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
```shell
# SSH
git clone https://github.com/AllenXiangX/VGGS.git
cd VGGS

uv venv --python 3.12
source .venv/bin/activate

uv sync

# Optional: install hyperparameter tuning helpers.
uv sync --extra tuning

# Install the PyTorch stack separately so it can match your CUDA/JetPack version.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # replace with your CUDA-compatible wheel index

# Install PyTorch3D separately. On Python 3.12 and Jetson/aarch64, expect to
# build it from source against the exact PyTorch/CUDA stack you installed.
uv pip install pytorch3d

uv pip install submodules/diff-plane-rasterization
uv pip install submodules/simple-knn
```

For Jetson AGX Orin / JetPack, install the NVIDIA-provided PyTorch build first
and see [Python 3.12 / Jetson notes](docs/python312-jetson.md).

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
