import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

INTERMEDIATE_SCENES = ["Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train"]
ADVANCED_SCENES = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]
IMAGE_SET_ARCHIVES = {
    "intermediate": "https://storage.googleapis.com/t2-downloads/image_sets/intermediate.zip",
    "advanced": "https://storage.googleapis.com/t2-downloads/image_sets/advanced.zip",
}


def run(cmd, cwd=None, dry_run=False):
    printable = " ".join(str(x) for x in cmd)
    print(printable, flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def scenes_for_group(group):
    if group == "intermediate":
        return INTERMEDIATE_SCENES
    if group == "advanced":
        return ADVANCED_SCENES
    if group == "both":
        return INTERMEDIATE_SCENES + ADVANCED_SCENES
    raise ValueError(group)


def ensure_toolbox(toolbox_dir, dry_run=False):
    toolbox_dir.mkdir(parents=True, exist_ok=True)
    downloader = toolbox_dir / "download_t2_dataset.py"
    uploader = toolbox_dir / "upload_t2_results.py"
    if downloader.exists() and uploader.exists():
        return
    repo = toolbox_dir / "TanksAndTemples"
    if not repo.exists():
        run(["git", "clone", "https://github.com/isl-org/TanksAndTemples.git", str(repo)], dry_run=dry_run)
    src = repo / "python_toolbox"
    for name in ["download_t2_dataset.py", "upload_t2_results.py", "convert_to_logfile.py", "interpolate_log_file.py"]:
        if not dry_run:
            shutil.copy2(src / name, toolbox_dir / name)


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5_for_archive(group):
    chk_url = IMAGE_SET_ARCHIVES[group].replace(".zip", ".chk")
    with urllib.request.urlopen(chk_url, timeout=30) as response:
        return response.read().decode("utf-8").split()[0]


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destination)


def download_images(toolbox_dir, group, dry_run=False):
    if group == "both":
        for name in ["intermediate", "advanced"]:
            download_images(toolbox_dir, name, dry_run=dry_run)
        return
    archive = toolbox_dir / "image_sets" / f"{group}.zip"
    if dry_run:
        print(f"download {IMAGE_SET_ARCHIVES[group]} -> {archive}")
        return
    if not archive.exists():
        download_file(IMAGE_SET_ARCHIVES[group], archive)
    expected = expected_md5_for_archive(group)
    actual = md5sum(archive)
    if actual != expected:
        raise RuntimeError(f"MD5 mismatch for {archive}: expected {expected}, got {actual}")
    print(f"Verified {archive} ({actual})")


def normalize_image_set(download_root, workspace_root, group, dry_run=False):
    if group == "both":
        for name in ["intermediate", "advanced"]:
            normalize_image_set(download_root, workspace_root, name, dry_run=dry_run)
        return
    scenes = set(scenes_for_group(group))
    archive = download_root / "image_sets" / f"{group}.zip"
    if dry_run:
        print(f"extract {archive} -> {workspace_root / group / '<Scene>' / 'images'}")
        return
    with ZipFile(archive) as zf:
        counts = {scene: 0 for scene in scenes}
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2 or parts[0] not in scenes:
                continue
            scene = parts[0]
            target = workspace_root / group / scene / "images" / parts[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                counts[scene] += 1
                continue
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[scene] += 1
    for scene in sorted(counts):
        print(f"{scene}: {counts[scene]} images")


def run_colmap_for_scene(scene_dir, colmap="colmap", camera_model="PINHOLE", use_gpu=False, dry_run=False):
    images = scene_dir / "images"
    sparse = scene_dir / "sparse"
    database = scene_dir / "database.db"
    if not images.exists():
        if dry_run:
            print(f"[dry-run] images not present yet: {images}")
        else:
            raise FileNotFoundError(images)
    if not dry_run:
        sparse.mkdir(parents=True, exist_ok=True)
    run([colmap, "feature_extractor", "--database_path", database, "--image_path", images,
         "--ImageReader.camera_model", camera_model, "--SiftExtraction.use_gpu", "1" if use_gpu else "0"], dry_run=dry_run)
    run([colmap, "sequential_matcher", "--database_path", database, "--SiftMatching.use_gpu", "1" if use_gpu else "0"], dry_run=dry_run)
    run([colmap, "mapper", "--database_path", database, "--image_path", images, "--output_path", sparse], dry_run=dry_run)

    first_model = sparse / "0"
    if not dry_run and not first_model.exists():
        raise RuntimeError(f"COLMAP did not create {first_model}")
    undistorted = scene_dir / "undistorted"
    run([colmap, "image_undistorter", "--image_path", images, "--input_path", first_model,
         "--output_path", undistorted, "--output_type", "COLMAP"], dry_run=dry_run)
    if not dry_run:
        final_sparse = scene_dir / "sparse"
        final_images = scene_dir / "images"
        undist_sparse = undistorted / "sparse"
        undist_images = undistorted / "images"
        if undist_sparse.exists():
            for item in undist_sparse.iterdir():
                target = final_sparse / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
        if undist_images.exists() and not any(final_images.glob("*.png")):
            shutil.rmtree(final_images)
            shutil.move(str(undist_images), str(final_images))


def make_log(toolbox_dir, scene_dir, scene_name, image_ext="jpg", dry_run=False):
    sparse = scene_dir / "sparse"
    model = sparse / "cameras.bin" if (sparse / "cameras.bin").exists() else sparse / "0" / "cameras.bin"
    out = scene_dir / f"{scene_name}.log"
    run([sys.executable, toolbox_dir / "convert_to_logfile.py", model, out, scene_dir / "images", "COLMAP", image_ext], dry_run=dry_run)


def check_vggt_priors(scene_dir):
    required = [scene_dir / "depth_vggt", scene_dir / "normal"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "VGGT priors are missing. VGGS needs depth_vggt/*_pred.npy, depth_vggt/*_conf.npy, "
            f"and normal/*_normal.npy before training. Missing: {missing}"
        )


def train_scene(scene_dir, output_root, scene_name, config, common_args, n_trials, dry_run=False):
    if not dry_run:
        check_vggt_priors(scene_dir)
    else:
        print(f"[dry-run] would require VGGT priors under {scene_dir / 'depth_vggt'} and {scene_dir / 'normal'}")
    model_root = output_root / scene_name
    cmd = [sys.executable, "train_optuna.py", "-s", scene_dir, "-m", model_root,
           "--config", config, "--n-trials", str(n_trials), "--metric-split", "train",
           "--storage", f"sqlite:///{model_root / (scene_name.lower() + '_optuna.db')}",
           "--study-name", f"vggs-tnt-official-{scene_name.lower()}",
           "--report-dir", model_root / "report", "--common-args", common_args]
    run(cmd, cwd=Path.cwd(), dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Official Tanks and Temples helper for VGGS submission prep.")
    parser.add_argument("--group", choices=["intermediate", "advanced", "both"], default="intermediate")
    parser.add_argument("--workspace-root", default="data/tnt_official")
    parser.add_argument("--toolbox-dir", default="tools/tanksandtemples")
    parser.add_argument("--output-root", default="exp/tnt_official")
    parser.add_argument("--config", default="configs/tnt_10views.yaml")
    parser.add_argument("--common-args", default="-r2 --ncc_scale 0.5 --data_device cuda --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--scene", help="Process only one scene from the selected group")
    parser.add_argument("--colmap-use-gpu", action="store_true", help="Enable COLMAP SIFT GPU. Leave off for Ubuntu arm64 COLMAP builds without CUDA.")
    parser.add_argument("--image-ext", default="jpg")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-colmap", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    toolbox_dir = Path(args.toolbox_dir)
    workspace_root = Path(args.workspace_root)
    if args.download:
        download_images(toolbox_dir, args.group, dry_run=args.dry_run)
        normalize_image_set(toolbox_dir, workspace_root, args.group, dry_run=args.dry_run)

    groups = ["intermediate", "advanced"] if args.group == "both" else [args.group]
    for group in groups:
        selected_scenes = scenes_for_group(group)
        if args.scene:
            selected_scenes = [scene for scene in selected_scenes if scene == args.scene]
            if not selected_scenes:
                print(f"Skipping group {group}: scene {args.scene} is not part of it")
                continue
        for scene in selected_scenes:
            scene_dir = workspace_root / group / scene
            if args.dry_run and not scene_dir.exists():
                print(f"[dry-run] would prepare scene directory: {scene_dir}")
            if args.prepare_colmap:
                if args.dry_run or (scene_dir / "images").exists():
                    run_colmap_for_scene(scene_dir, colmap=args.colmap, use_gpu=args.colmap_use_gpu, dry_run=args.dry_run)
                    make_log(toolbox_dir, scene_dir, scene, image_ext=args.image_ext, dry_run=args.dry_run)
                else:
                    print(f"Skipping {scene}: missing images at {scene_dir / 'images'}")
            if args.train:
                if args.dry_run or scene_dir.exists():
                    train_scene(scene_dir, Path(args.output_root) / group, scene, args.config, args.common_args, args.n_trials, dry_run=args.dry_run)
                else:
                    print(f"Skipping {scene}: missing scene directory {scene_dir}")


if __name__ == "__main__":
    main()
