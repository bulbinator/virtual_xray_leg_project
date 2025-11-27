"""
X-ray forward model using a simple parallel-beam approximation
and Beer–Lambert law for a 3D leg phantom.

Author: (your name)
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import rotate, gaussian_filter


def project_parallel_beam(
    mu_slice: np.ndarray,
    angle_deg: float = 0.0,
    voxel_size_mm: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a parallel-beam projection for a single 2D μ-map.

    Steps:
    1. Rotate μ-map by the given angle (around its center).
    2. Sum along columns to obtain line integrals.

    Parameters
    ----------
    mu_slice : np.ndarray
        2D slice of μ values (Ny, Nx).
    angle_deg : float
        Projection angle in degrees (counterclockwise).
    voxel_size_mm : float
        Physical size of each voxel in mm.

    Returns
    -------
    line_integrals : np.ndarray
        1D array of line integrals (per detector pixel).
    mu_rotated : np.ndarray
        Rotated μ-map used for the projection.
    """
    mu_rot = rotate(
        mu_slice,
        angle=angle_deg,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )

    # Sum along rows (axis=0) to get line integrals along each column.
    line_integrals = np.sum(mu_rot, axis=0) * voxel_size_mm
    return line_integrals.astype(np.float32), mu_rot.astype(np.float32)


def simulate_projection(
    phantom: np.ndarray,
    angle_deg: float = 0.0,
    kVp: float = 70.0,
    distance_cm: float = 100.0,
    I0: float = 1.0,
    noise_level: float = 0.0,
    blur_sigma: float = 0.0,
    voxel_size_mm: float = 1.0,
    kVp_ref: float = 70.0,
    dist_ref_cm: float = 100.0,
    add_poisson_noise: bool = True,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a 2D X-ray projection image from a 3D phantom.

    Model:
    - Parallel beam (no divergence).
    - Beer–Lambert: I = I0 * exp(-∑ μ * dx).
    - Effective attenuation scales with kVp (simple 1/kVp relationship).
    - Distance affects intensity via inverse-square (1/d^2).
    - Optional Poisson noise and Gaussian blur (system PSF).

    Parameters
    ----------
    phantom : np.ndarray
        3D μ-map (Nz, Ny, Nx).
    angle_deg : float
        Projection angle in degrees (for the 2D central slice).
    kVp : float
        Tube potential (kVp). Higher kVp → lower effective attenuation.
    distance_cm : float
        Source–detector distance in cm (affects geometric intensity).
    I0 : float
        Incident intensity (arbitrary units).
    noise_level : float
        Noise level in [0, 1]. 0 = no extra noise.
    blur_sigma : float
        Gaussian blur standard deviation (in pixels). 0 = no blur.
    voxel_size_mm : float
        Voxel size in mm for line integral computation.
    kVp_ref : float
        Reference kVp used in the attenuation scaling.
    dist_ref_cm : float
        Reference distance for intensity scaling.
    add_poisson_noise : bool
        If True, use Poisson noise; otherwise Gaussian.
    random_state : int | None
        Seed for NumPy random number generator.

    Returns
    -------
    proj_image : np.ndarray
        Final noisy/blurred projection image (Ny, Nx_proj).
    proj_clean : np.ndarray
        Clean projection (no noise, no blur) with same size.
    mu_slice : np.ndarray
        Central μ slice used.
    mu_rot : np.ndarray
        Rotated μ slice.
    line_integrals : np.ndarray
        1D line integrals before exponentiation.
    """
    rng = np.random.default_rng(random_state)

    Nz, Ny, Nx = phantom.shape
    mu_slice = phantom[Nz // 2, :, :].astype(np.float32)

    # Simple effective-energy model:
    # higher kVp -> lower effective attenuation.
    mu_eff = mu_slice * (kVp_ref / float(kVp))

    line_integrals, mu_rot = project_parallel_beam(
        mu_eff, angle_deg=angle_deg, voxel_size_mm=voxel_size_mm
    )

    # Beer–Lambert with inverse-square distance scaling.
    geom_factor = (dist_ref_cm / float(distance_cm)) ** 2
    I_clean_1d = I0 * geom_factor * np.exp(-line_integrals)

    # Form a 2D image by repeating the 1D profile along rows.
    proj_clean = np.tile(I_clean_1d[np.newaxis, :], (Ny, 1)).astype(np.float32)

    proj_noisy = proj_clean.copy()

    # Add noise
    if noise_level > 0:
        if add_poisson_noise:
            # Map noise_level [0,1] to photon counts:
            max_photons = 5e5
            min_photons = 5e3
            N0 = max_photons - noise_level * (max_photons - min_photons)
            expected_counts = proj_clean * N0
            noisy_counts = rng.poisson(expected_counts)
            proj_noisy = noisy_counts.astype(np.float32) / float(N0)
        else:
            # Gaussian approximation
            sigma = noise_level * 0.05 * float(np.max(proj_clean))
            noise = rng.normal(loc=0.0, scale=sigma, size=proj_clean.shape)
            proj_noisy = proj_clean + noise

        proj_noisy = np.clip(proj_noisy, 0.0, None)

    # Add blur (system PSF)
    if blur_sigma > 0:
        proj_blur = gaussian_filter(proj_noisy, sigma=blur_sigma)
    else:
        proj_blur = proj_noisy

    proj_blur = proj_blur.astype(np.float32)
    proj_clean = proj_clean.astype(np.float32)

    return proj_blur, proj_clean, mu_slice, mu_rot, line_integrals


if __name__ == "__main__":
    # Minimal smoke test.
    from phantom.generate_phantom import create_default_leg_phantom
    import matplotlib.pyplot as plt

    ph = create_default_leg_phantom()
    proj, proj_clean, mu_slice, mu_rot, L = simulate_projection(
        ph, angle_deg=30.0, kVp=70.0, distance_cm=100.0, noise_level=0.2, blur_sigma=1.0
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(proj, cmap="gray", origin="lower")
    plt.title("Example simulated X-ray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
