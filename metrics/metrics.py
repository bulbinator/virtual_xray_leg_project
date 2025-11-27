import numpy as np
from scipy.ndimage import sobel, zoom
from typing import Optional

try:
    # Optional: use skimage's SSIM if available
    from skimage.metrics import structural_similarity as skimage_ssim  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    skimage_ssim = None


def _ensure_float(image: np.ndarray) -> np.ndarray:
    """Convert image to float64 numpy array."""
    return np.asarray(image, dtype=np.float64)


def _resize_to_match(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """
    Resize `moving` image to match `reference` shape using bilinear interpolation.

    If shapes already match, returns `moving` unchanged.
    """
    if reference.shape == moving.shape:
        return moving
    zoom_factors = [r / float(m) for r, m in zip(reference.shape, moving.shape)]
    return zoom(moving, zoom=zoom_factors, order=1)


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Mean Squared Error (MSE) between two images."""
    x = _ensure_float(image1)
    y = _ensure_float(image2)
    y_resized = _resize_to_match(x, y)
    diff = x - y_resized
    return float(np.mean(diff**2))


def ssim(image1: np.ndarray, image2: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Structural Similarity Index (SSIM) between two images.

    Uses skimage.metrics.structural_similarity if available; otherwise falls back
    to a simple global SSIM approximation.
    """
    x = _ensure_float(image1)
    y = _ensure_float(image2)
    y_resized = _resize_to_match(x, y)

    if data_range is None:
        data_range = float(x.max() - x.min()) or 1.0

    if skimage_ssim is not None:
        # skimage expects 2D arrays
        return float(skimage_ssim(x, y_resized, data_range=data_range))

    # Fallback: global SSIM approximation (no sliding window)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y_resized)
    sigma_x2 = np.var(x)
    sigma_y2 = np.var(y_resized)
    sigma_xy = np.mean((x - mu_x) * (y_resized - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel filters."""
    x = _ensure_float(image)
    gx = sobel(x, axis=1)  # horizontal gradient
    gy = sobel(x, axis=0)  # vertical gradient
    return np.sqrt(gx**2 + gy**2)


def gradient_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Gradient error (GE) between two images.

    Defined here as the mean absolute difference between gradient magnitudes.
    """
    g1 = gradient_magnitude(image1)
    g2 = gradient_magnitude(image2)
    g2_resized = _resize_to_match(g1, g2)
    return float(np.mean(np.abs(g1 - g2_resized)))


def gradient_entropy(image: np.ndarray, num_bins: int = 64) -> float:
    """
    Gradient entropy (gEn) of an image.

    1. Compute gradient magnitude G.
    2. Build histogram of G (num_bins bins).
    3. Normalize to probabilities p(i).
    4. Compute entropy: gEn = -sum p(i) log2 p(i).

    Higher gEn corresponds to richer edge content.
    """
    G = gradient_magnitude(image).ravel()
    if G.size == 0:
        return 0.0
    g_min, g_max = float(G.min()), float(G.max())
    if g_max <= g_min:
        return 0.0
    hist, bin_edges = np.histogram(G, bins=num_bins, range=(g_min, g_max), density=False)
    total = float(hist.sum())
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)


def compute_all_metrics(
    reference: np.ndarray,
    test: np.ndarray,
    num_bins: int = 64,
) -> dict:
    """
    Compute a set of image quality metrics between a reference and a test image.

    Metrics:
        - MSE
        - SSIM
        - Gradient Error (GE)
        - Gradient Entropy (gEn) for reference and test

    Returns
    -------
    metrics : dict
        Dictionary with metric names as keys.
    """
    ref = _ensure_float(reference)
    tst = _ensure_float(test)
    mse_val = mse(ref, tst)
    ssim_val = ssim(ref, tst)
    ge_val = gradient_error(ref, tst)
    gen_ref = gradient_entropy(ref, num_bins=num_bins)
    gen_test = gradient_entropy(tst, num_bins=num_bins)

    return {
        "MSE": mse_val,
        "SSIM": ssim_val,
        "GradientError": ge_val,
        "GradientEntropy_ref": gen_ref,
        "GradientEntropy_test": gen_test,
    }
