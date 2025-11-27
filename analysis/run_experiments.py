import csv
import os

import numpy as np

from phantom import generate_leg_phantom
from xray_simulator import simulate_projection
from metrics.metrics import compute_all_metrics

# Parameter ranges
ANGLES_DEG = [0, 30, 60, 90]
KVP_VALUES = [50, 60, 70, 80, 90]
DISTANCES_CM = [80, 100, 120]
NOISE_LEVELS = [0.0, 0.3, 0.6]
BLUR_SIGMAS = [0.0, 1.0, 2.0]


def run_experiments() -> None:
    np.random.seed(0)  # for reproducibility

    # Output folders
    results_dir = "results"
    images_dir = os.path.join(results_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Generate a fixed phantom
    phantom = generate_leg_phantom()

    # Precompute reference images for each angle
    reference_images: dict[int, np.ndarray] = {}
    for angle in ANGLES_DEG:
        reference_images[angle] = simulate_projection(
            phantom,
            angle_deg=angle,
            kVp=70.0,
            distance_cm=100.0,
            noise_level=0.0,
            blur_sigma=0.0,
        )

    # CSV file for metrics
    csv_path = os.path.join(results_dir, "metrics.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "angle_deg",
                "kVp",
                "distance_cm",
                "noise_level",
                "blur_sigma",
                "MSE",
                "SSIM",
                "GradientError",
                "GradientEntropy_ref",
                "GradientEntropy_test",
                "image_path",
            ]
        )

        # Parameter sweep
        for angle in ANGLES_DEG:
            ref_img = reference_images[angle]
            for kvp in KVP_VALUES:
                for distance in DISTANCES_CM:
                    for noise in NOISE_LEVELS:
                        for blur in BLUR_SIGMAS:
                            test_img = simulate_projection(
                                phantom,
                                angle_deg=angle,
                                kVp=kvp,
                                distance_cm=distance,
                                noise_level=noise,
                                blur_sigma=blur,
                            )

                            metrics = compute_all_metrics(ref_img, test_img)

                            # Save test image as .npy
                            img_filename = (
                                f"proj_angle{angle}_kvp{kvp}_dist{distance}"
                                f"_noise{noise}_blur{blur}.npy"
                            )
                            img_path = os.path.join(images_dir, img_filename)
                            np.save(img_path, test_img)

                            writer.writerow(
                                [
                                    angle,
                                    kvp,
                                    distance,
                                    noise,
                                    blur,
                                    metrics["MSE"],
                                    metrics["SSIM"],
                                    metrics["GradientError"],
                                    metrics["GradientEntropy_ref"],
                                    metrics["GradientEntropy_test"],
                                    img_path,
                                ]
                            )

    print(f"Experiments complete. Metrics saved to: {csv_path}")


if __name__ == "__main__":
    run_experiments()
