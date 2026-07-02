# Python 3.12 / Jetson AGX Orin Notes

This repo was originally documented with Python 3.8 and CUDA 11.8. The Python
code parses under Python 3.12, but the CUDA/PyTorch dependency stack must be
installed carefully.

## Recommended order

1. Start from an NVIDIA JetPack-compatible container or system image.
2. Create the Python environment with uv:

   ```shell
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync
   ```

3. Install the PyTorch, TorchVision and TorchAudio builds that match JetPack.
   Prefer NVIDIA's Jetson wheels/containers over upstream PyPI wheels.
4. Build PyTorch3D from source if no matching Jetson/aarch64 wheel exists.
5. Build the local CUDA extensions with uv:

   ```shell
   export TORCH_CUDA_ARCH_LIST="8.7"
   uv pip install -v submodules/diff-plane-rasterization
   uv pip install -v submodules/simple-knn
   ```

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
