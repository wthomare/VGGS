# Python 3.12 / Jetson AGX Orin Notes

This repo was originally documented with Python 3.8 and CUDA 11.8. The Python
code parses under Python 3.12, but the CUDA/PyTorch dependency stack must be
installed carefully.

## Recommended order

1. Start from an NVIDIA JetPack-compatible container or system image.

2. Create the Python environment with uv:

   ```shell
   cd ~/Desktop/VGGS
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync --extra tuning
   ```

   `open3d` is not part of the base Jetson install because PyPI does not ship
   Linux aarch64 wheels for Open3D. `lpips` is also optional here because the
   PyPI package can pull a generic desktop PyTorch stack.

3. Install the PyTorch, TorchVision and TorchAudio builds that match JetPack.
   Prefer NVIDIA's Jetson wheels/containers over upstream PyPI wheels. Do not
   use `https://download.pytorch.org/whl/cu126` on Jetson; that index targets
   desktop/server Linux wheels, not Jetson aarch64.

   Verify the installed stack:

   ```shell
   python - <<'PY'
   import torch
   print(torch.__version__)
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   PY
   ```

4. Build PyTorch3D from source if no matching Jetson/aarch64 wheel exists:

   ```shell
   export CUDA_HOME=/usr/local/cuda-12.6
   export PATH="$CUDA_HOME/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
   export TORCH_CUDA_ARCH_LIST="8.7"
   export MAX_JOBS=2

   uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
   ```

5. Build the local CUDA extensions with uv:

   ```shell
   export CUDA_HOME=/usr/local/cuda-12.6
   export PATH="$CUDA_HOME/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
   export TORCH_CUDA_ARCH_LIST="8.7"
   export MAX_JOBS=2

   uv pip install -v --no-build-isolation submodules/diff-plane-rasterization
   uv pip install -v --no-build-isolation submodules/simple-knn
   ```

6. Run a small Optuna job or a direct train command:

   ```shell
   python train_optuna.py \
     -s data/DTU/set_22_25_28/scan24/dense \
     -m exp/optuna/dtu_scan24 \
     --config configs/dtu.yaml \
     --n-trials 20 \
     --test-iteration 3000 \
     --metric-split test \
     --sampler tpe \
     --pruner none \
     --storage sqlite:///exp/optuna/dtu_scan24.db \
     --report-dir exp/optuna/dtu_scan24_report \
     --common-args "--quiet -r2 --ncc_scale 0.5"
   ```

## Common Jetson errors

- `ModuleNotFoundError: No module named 'torch'` while building a CUDA
  submodule: rerun the install with `--no-build-isolation`.
- `detected CUDA version ... mismatches ... PyTorch`: replace PyTorch with the
  NVIDIA Jetson wheel matching your JetPack/CUDA version.
- `No published PyTorch CUDA builds ... support GPU0 Orin CC 8.7`: the PyTorch
  build is not suitable for AGX Orin; install a Jetson-compatible NVIDIA build.
- `identifier FLT_MAX is undefined` in `simple_knn.cu`: ensure this fork's
  `#include <cfloat>` patch is present in `submodules/simple-knn/simple_knn.cu`.
- `sqlite3.OperationalError: unable to open database file` in Optuna: create the
  parent directory for the SQLite file, or use this fork's patched
  `train_optuna.py`, which creates it automatically.

## Notes

- Jetson AGX Orin uses unified memory, not dedicated desktop-style VRAM. Leave
  memory headroom for the OS and dataset preprocessing.
- Keep `--data_device cpu` in mind for very large scenes if image preloading
  becomes the memory bottleneck.
- The training configs expose the main quality/stability knobs:
  `depth_conf_keep_ratio`, `depth_conf_keep_ratio_2`, `loss_ramp_iters`,
  `robust_depth_align`, `depth_align_irls_iters`, and
  `depth_align_min_anchors`, `depth_edge_keep_ratio`, and `pseudo_depth_ema`.
- Use `scripts/sweep_quality.py` for small ablation sweeps before committing to
  longer Jetson runs.
- Install `uv sync --extra tuning` only when running Optuna sweeps; it is kept
  optional so the base Jetson environment stays smaller.
- If a CUDA extension build fails, first verify that `python -c "import torch;
  print(torch.__version__, torch.version.cuda)"` matches the JetPack CUDA stack.
