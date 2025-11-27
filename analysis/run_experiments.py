"""
Script for structured experiments with the virtual X-ray leg phantom.

It:
1. Generates a fixed phantom.
2. Loops over:
   - Angles: 0, 30, 60, 90 deg
   - kVp: 50, 70, 90
   - Distances: 80, 100, 120 cm
   - Noise levels: 0.0, 0.2, 0.5
3. For each configuration:
   - Computes a reference image (angle-specific, high-quality).
   - Computes a test image.
   - Calculates metrics: MSE, SSIM, GE, gEn_ref, gEn_test, VarLap_ref, VarLap_test.
   - Logs them to CSV.
   - Saves the test projection as a PNG.

Author: (your name)
"""

import csv
import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from phantom.generate_phantom import create_default_leg_phantom
from simulation.xray_projection import simulate_projection
from metrics.metrics import compute_all_metrics


# Reference parameters
REF_KVP = 70.0
REF_DISTANCE_CM = 100.0
REF_NOISE = 0.0
REF_BLUR = 0.0
REF_I0 = 1.0


def run_experiments():
    # Output paths
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "experiment_results.csv")

    # Phantom
    phantom = create_default_leg_phantom()

    # Parameter grid
    angles = [0.0, 30.0, 60.0, 90.0]
    kvps = [50.0, 70.0, 90.0]
    distances = [80.0, 100.0, 120.0]
    noise_levels = [0.0, 0.2, 0.5]
    blur_sigma = 1.0  # fixed blur for all, could also be varied

    rows = []

    for angle, kvp, distance, noise in product(angles, kvps, distances, noise_levels):
        print(
            f"Running angle={angle}°, kVp={kvp}, distance={distance} cm, noise={noise}..."
        )

        # Reference image for this angle
        ref_img, _, _, _, _ = simulate_projection(
            phantom,
            angle_deg=angle,
            kVp=REF_KVP,
            distance_cm=REF_DISTANCE_CM,
            I0=REF_I0,
            noise_level=REF_NOISE,
            blur_sigma=REF_BLUR,
        )

        # Test image
        test_img, _, _, _, _ = simulate_projection(
            phantom,
            angle_deg=angle,
            kVp=kvp,
            distance_cm=distance,
            I0=1.0,
            noise_level=noise,
            blur_sigma=blur_sigma,
        )

        # Metrics
        metrics = compute_all_metrics(ref_img, test_img)

        row = {
            "angle_deg": angle,
            "kVp": kvp,
            "distance_cm": distance,
            "noise_level": noise,
            "blur_sigma": blur_sigma,
        }
        row.update(metrics)
        rows.append(row)

        # Save image
        img_filename = (
            f"proj_angle{int(angle)}_kvp{int(kvp)}"
            f"_dist{int(distance)}_noise{int(noise*100)}.png"
        )
        img_path = os.path.join(results_dir, img_filename)
        plt.imsave(img_path, test_img, cmap="gray")

    # Write CSV
    fieldnames = [
        "angle_deg",
        "kVp",
        "distance_cm",
        "noise_level",
        "blur_sigma",
        "MSE",
        "SSIM",
        "GE",
        "gEn_ref",
        "gEn_test",
        "VarLap_ref",
        "VarLap_test",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved metrics to: {csv_path}")
    print(f"Saved images in: {results_dir}")


if __name__ == "__main__":
    run_experiments()
