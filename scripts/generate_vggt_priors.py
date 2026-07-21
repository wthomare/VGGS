import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def image_files(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in Path(image_dir).iterdir() if p.suffix in exts])


def normals_from_points(points):
    # points: H, W, 3, in a common 3D coordinate frame. Normals are oriented to -Z-ish
    pts = points.astype(np.float32)
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dx[:, 0] = pts[:, 1] - pts[:, 0]
    dx[:, -1] = pts[:, -1] - pts[:, -2]
    dy[1:-1] = pts[2:] - pts[:-2]
    dy[0] = pts[1] - pts[0]
    dy[-1] = pts[-1] - pts[-2]
    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-8)
    # Keep a stable convention close to the existing VGGS priors.
    flip = normal[..., 2:3] > 0
    normal = np.where(flip, -normal, normal)
    normal[~np.isfinite(normal)] = 0
    return normal.astype(np.float32)


def crop_resize_tensor(tensor, coord, target_hw, kind):
    # tensor can be H,W or H,W,C from VGGT square prediction.
    if kind in {"depth", "conf"} and tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    h, w = tensor.shape[:2]
    x1, y1, x2, y2, orig_w, orig_h = coord.tolist()
    scale_x = w / 1024.0
    scale_y = h / 1024.0
    ix1 = max(0, int(round(x1 * scale_x)))
    iy1 = max(0, int(round(y1 * scale_y)))
    ix2 = min(w, int(round(x2 * scale_x)))
    iy2 = min(h, int(round(y2 * scale_y)))
    cropped = tensor[iy1:iy2, ix1:ix2]
    if cropped.ndim == 2:
        t = torch.from_numpy(cropped)[None, None]
    else:
        t = torch.from_numpy(cropped).permute(2, 0, 1)[None]
    mode = "nearest" if kind == "normal" else "bilinear"
    resized = F.interpolate(t.float(), size=target_hw, mode=mode, align_corners=False if mode == "bilinear" else None)
    if cropped.ndim == 2:
        return resized[0, 0].cpu().numpy().astype(np.float32)
    return resized[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def load_model(device):
    from vggt.models.vggt import VGGT
    model = None
    if hasattr(VGGT, "from_pretrained"):
        try:
            model = VGGT.from_pretrained("facebook/VGGT-1B")
        except Exception as exc:
            print(f"VGGT.from_pretrained failed, falling back to model.pt: {exc}")
    if model is None:
        model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state)
    return model.eval().to(device)


def run_batch(model, paths, args, device, dtype):
    from PIL import Image
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    images, original_coords = load_and_preprocess_images_square([str(p) for p in paths], args.load_resolution)
    original_coords_np = original_coords.cpu().numpy()
    images = images.to(device)
    images_model = F.interpolate(images, size=(args.vggt_resolution, args.vggt_resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        autocast_ctx = torch.amp.autocast("cuda", dtype=dtype) if device == "cuda" else torch.amp.autocast("cpu", enabled=False)
        with autocast_ctx:
            batch = images_model[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch, ps_idx)

    depth_np = depth_map.squeeze(0).detach().float().cpu().numpy()
    conf_np = depth_conf.squeeze(0).detach().float().cpu().numpy()
    points_np = unproject_depth_map_to_point_map(
        depth_np,
        extrinsic.squeeze(0).detach().float().cpu().numpy(),
        intrinsic.squeeze(0).detach().float().cpu().numpy(),
    )

    for i, path in enumerate(paths):
        with Image.open(path) as im:
            orig_w, orig_h = im.size
        target_h = max(1, int(round(orig_h / args.resolution_factor)))
        target_w = max(1, int(round(orig_w / args.resolution_factor)))
        stem = path.stem
        depth = crop_resize_tensor(depth_np[i], original_coords_np[i], (target_h, target_w), "depth")
        conf = crop_resize_tensor(conf_np[i], original_coords_np[i], (target_h, target_w), "conf")
        points = crop_resize_tensor(points_np[i], original_coords_np[i], (target_h, target_w), "points")
        normal = normals_from_points(points)

        np.save(args.scene_dir / "depth_vggt" / f"{stem}_pred.npy", depth)
        np.save(args.scene_dir / "depth_vggt" / f"{stem}_conf.npy", conf)
        np.save(args.scene_dir / "normal" / f"{stem}_normal.npy", normal)
        print(f"saved priors for {path.name}: depth {depth.shape}, normal {normal.shape}", flush=True)

    del images, images_model, depth_map, depth_conf, depth_np, conf_np, points_np
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Generate VGGT depth/confidence/normal priors for VGGS scenes.")
    parser.add_argument("scene_dir", type=Path, help="Scene directory containing images/")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of images per VGGT pass. Lower this on Jetson if memory is tight.")
    parser.add_argument("--resolution-factor", type=float, default=2.0, help="Match VGGS -r value; -r2 means output half-size priors.")
    parser.add_argument("--load-resolution", type=int, default=1024)
    parser.add_argument("--vggt-resolution", type=int, default=518)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images for a smoke test.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.scene_dir = args.scene_dir.resolve()
    image_dir = args.scene_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    (args.scene_dir / "depth_vggt").mkdir(exist_ok=True)
    (args.scene_dir / "normal").mkdir(exist_ok=True)

    paths = image_files(image_dir)
    if args.limit:
        paths = paths[: args.limit]
    if args.skip_existing:
        paths = [p for p in paths if not (args.scene_dir / "depth_vggt" / f"{p.stem}_pred.npy").exists()]
    if not paths:
        print("No images to process.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for practical VGGT prior generation on Jetson.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using {device}, dtype={dtype}, {len(paths)} images", flush=True)
    model = load_model(device)

    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        print(f"Batch {start // args.batch_size + 1}: {batch_paths[0].name}..{batch_paths[-1].name}", flush=True)
        run_batch(model, batch_paths, args, device, dtype)


if __name__ == "__main__":
    main()
