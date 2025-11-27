"""
Image quality metrics for X-ray simulation.

Includes:
- MSE
- SSIM
- Gradient Error
- Gradient Entropy (gEn)
- Variance of Laplacian (optional sharpness metric)

Author: (your name)
"""

from typing import Dict

import numpy as np
from scipy.ndimage import sobel, laplace
from skimage.metrics import structural_similarity as ssim


def _to_float32(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Mean Squared Error between two images.
    """
    a = _to_float32(img1)
    b = _to_float32(img2)
    assert a.shape == b.shape, "Images must have the same shape for MSE."
    diff = a - b
    return float(np.mean(diff**2))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index Measure between two images.
    """
    a = _to_float32(img1)
    b = _to_float32(img2)
    assert a.shape == b.shape, "Images must have the same shape for SSIM."
    data_range = float(np.max(b) - np.min(b)) or 1.0
    val = ssim(a, b, data_range=data_range, channel_axis=None)
    return float(val)


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude using Sobel filters.
    """
    img = _to_float32(img)
    gx = sobel(img, axis=1)  # x-direction
    gy = sobel(img, axis=0)  # y-direction
    return np.sqrt(gx**2 + gy**2)


def compute_gradient_error(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Gradient Error (GE): mean absolute difference of gradient magnitudes.
    """
    g1 = _gradient_magnitude(img1)
    g2 = _gradient_magnitude(img2)
    assert g1.shape == g2.shape, "Gradient maps must have the same shape."
    return float(np.mean(np.abs(g1 - g2)))


def compute_gradient_entropy(img: np.ndarray, n_bins: int = 64) -> float:
    """
    Gradient Entropy (gEn) for a single image.

    Steps:
    1. Compute gradient magnitude G.
    2. Build histogram (n_bins).
    3. Normalize histogram to probabilities p(i).
    4. Compute entropy: gEn = -sum p(i) log2 p(i).
    """
    G = _gradient_magnitude(img)
    G_flat = G.ravel().astype(np.float32)

    if G_flat.size == 0:
        return 0.0

    max_val = float(np.max(G_flat))
    if max_val == 0.0:
        return 0.0

    counts, _ = np.histogram(G_flat, bins=n_bins, range=(0.0, max_val), density=False)
    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts.astype(np.float64) / float(total)
    p = p[p > 0]  # avoid log(0)
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)


def compute_variance_of_laplacian(img: np.ndarray) -> float:
    """
    Optional sharpness metric: Variance of Laplacian.
    """
    img = _to_float32(img)
    lap = laplace(img)
    return float(np.var(lap))


def compute_all_metrics(ref: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    """
    Convenience function computing all key metrics at once.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (high-quality).
    test : np.ndarray
        Test image to be evaluated.

    Returns
    -------
    metrics : dict
        Dictionary with MSE, SSIM, GE, gEn_ref, gEn_test, VarLap_ref, VarLap_test.
    """
    mse_val = compute_mse(ref, test)
    ssim_val = compute_ssim(ref, test)
    ge_val = compute_gradient_error(ref, test)
    gen_ref = compute_gradient_entropy(ref)
    gen_test = compute_gradient_entropy(test)
    varlap_ref = compute_variance_of_laplacian(ref)
    varlap_test = compute_variance_of_laplacian(test)

    return {
        "MSE": mse_val,
        "SSIM": ssim_val,
        "GE": ge_val,
        "gEn_ref": gen_ref,
        "gEn_test": gen_test,
        "VarLap_ref": varlap_ref,
        "VarLap_test": varlap_test,
    }


if __name__ == "__main__":
    # Tiny sanity check: comparing image to itself.
    img = np.random.rand(64, 64).astype(np.float32)
    metrics = compute_all_metrics(img, img)
    print("Sanity check metrics (image vs itself):")
    for k, v in metrics.items():
        print(f"{k}: {v}")
