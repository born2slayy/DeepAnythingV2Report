import csv
import time
from pathlib import Path
from statistics import mean

import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


DEVICE = "cuda"
INPUT_SIZE = 518
RESULTS_DIR = Path("results")

CATEGORIES = [
    "asaken",
    "indoor",
    "outdoor",
    "mirror",
    "glass_window",
    "shiny_metal",
    "plain_wall",
    "very_close_object",
    "dark_image",
]

MODELS = [
    (
        "Small",
        "vits",
        {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        Path("checkpoints/depth_anything_v2_vits.pth"),
    ),
    (
        "Base",
        "vitb",
        {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        Path("checkpoints/depth_anything_v2_vitb.pth"),
    ),
    (
        "Large",
        "vitl",
        {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        Path("checkpoints/depth_anything_v2_vitl.pth"),
    ),
]


def colorize_depth(depth):
    depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-8)
    depth = (depth * 255).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    return (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not Path("photos").exists() or not Path("checkpoints").exists():
        raise RuntimeError("Run this script from the repository root.")

    photos = []
    for category in CATEGORIES:
        folder = Path("photos") / category
        if not folder.exists():
            raise FileNotFoundError(folder)

        source = "asaken" if category == "asaken" else "teammate"
        for photo in sorted(folder.iterdir()):
            if photo.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                photos.append((category, source, photo))

    rows = []
    total_jobs = len(MODELS) * len(photos)
    job = 0

    for model_name, encoder, config, checkpoint in MODELS:
        print(f"\nLoading {model_name} on {DEVICE}")

        model = DepthAnythingV2(**config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model = model.to(DEVICE).eval()

        depth_dir = RESULTS_DIR / model_name.lower() / "depth_maps"
        comparison_dir = RESULTS_DIR / model_name.lower() / "comparisons"
        depth_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)

        for category, source, photo in photos:
            job += 1
            image = cv2.imread(str(photo))
            if image is None:
                raise FileNotFoundError(photo)

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                depth = model.infer_image(image, INPUT_SIZE)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start

            depth_image = colorize_depth(depth)
            output_name = f"{category}_{photo.stem}_{model_name.lower()}.png"
            depth_path = depth_dir / output_name
            comparison_path = comparison_dir / output_name

            separator = np.full((image.shape[0], 30, 3), 255, dtype=np.uint8)
            comparison = cv2.hconcat([image, separator, depth_image])
            cv2.imwrite(str(depth_path), depth_image)
            cv2.imwrite(str(comparison_path), comparison)

            depth_min = float(depth.min())
            depth_max = float(depth.max())
            p05, p50, p95 = np.percentile(depth, [5, 50, 95])

            rows.append(
                {
                    "category": category,
                    "source": source,
                    "photo": photo.as_posix(),
                    "model": model_name,
                    "encoder": encoder,
                    "device": DEVICE,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "inference_time_sec": f"{inference_time:.6f}",
                    "depth_min": f"{depth_min:.6f}",
                    "depth_max": f"{depth_max:.6f}",
                    "depth_mean": f"{float(depth.mean()):.6f}",
                    "depth_std": f"{float(depth.std()):.6f}",
                    "depth_p05": f"{float(p05):.6f}",
                    "depth_p50": f"{float(p50):.6f}",
                    "depth_p95": f"{float(p95):.6f}",
                    "depth_range": f"{depth_max - depth_min:.6f}",
                    "depth_map": depth_path.as_posix(),
                    "comparison": comparison_path.as_posix(),
                }
            )

            print(f"[{job}/{total_jobs}] {model_name:<5} {category}/{photo.name} {inference_time:.2f}s")

        del model
        torch.cuda.empty_cache()

    with (RESULTS_DIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for model_name, encoder, _, _ in MODELS:
        model_rows = [row for row in rows if row["model"] == model_name]
        summary_rows.append(
            {
                "model": model_name,
                "encoder": encoder,
                "device": DEVICE,
                "runs": len(model_rows),
                "avg_inference_time_sec": f"{mean(float(row['inference_time_sec']) for row in model_rows):.6f}",
                "avg_depth_std": f"{mean(float(row['depth_std']) for row in model_rows):.6f}",
                "avg_depth_range": f"{mean(float(row['depth_range']) for row in model_rows):.6f}",
            }
        )

    with (RESULTS_DIR / "summary_by_model.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved results/results.csv")
    print("Saved results/summary_by_model.csv")


if __name__ == "__main__":
    main()
