import numpy as np
from scipy.ndimage import rotate, gaussian_filter, zoom

def simulate_projection(
    phantom: np.ndarray,
    angle_deg: float = 0.0,
    kVp: float = 70.0,
    distance_cm: float = 100.0,
    I0: float = 1e5,
    noise_level: float = 0.0,
    blur_sigma: float = 0.0,
    kVp_ref: float = 70.0,
    distance_ref_cm: float = 100.0,
    voxel_size_mm: float = 1.0,
) -> np.ndarray:
    """
    Simulate a 2D X-ray projection of a 3D phantom using a parallel-beam model.

    Parameters
    ----------
    phantom : np.ndarray
        3D array (Nz, Ny, Nx) of linear attenuation coefficients μ (1/cm).
    angle_deg : float
        Projection angle in degrees (rotation in the x-y plane).
    kVp : float
        Tube peak voltage (kVp). Used to scale effective μ.
    distance_cm : float
        Source–detector distance (arbitrary units, cm). Affects magnification and
        relative photon flux (1 / distance^2).
    I0 : float
        Baseline number of photons per detector pixel at the reference distance.
    noise_level : float
        Between 0 and 1. 0 = noise-free, 1 = high noise. Controls effective
        photon counts for the Poisson model.
    blur_sigma : float
        Standard deviation of Gaussian blur (in detector pixels).
    kVp_ref : float
        Reference kVp used for μ scaling.
    distance_ref_cm : float
        Reference distance used for magnification and flux.
    voxel_size_mm : float
        Isotropic voxel size in mm (used to convert to cm for integration).

    Returns
    -------
    image : np.ndarray
        2D projection image with values approximately in [0, 1].
    """
    if phantom.ndim != 3:
        raise ValueError("phantom must be a 3D array (Nz, Ny, Nx)")
    if kVp <= 0:
        raise ValueError("kVp must be positive")
    if distance_cm <= 0:
        raise ValueError("distance_cm must be positive")
    if I0 <= 0:
        raise ValueError("I0 must be positive")

    Nz, Ny, Nx = phantom.shape
    # Take central slice orthogonal to the cylinder axis
    slice_mu = phantom[Nz // 2, :, :].astype(np.float64)

    # Distance-dependent magnification (simple hack: larger distance -> less magnification)
    magnification = distance_ref_cm / float(distance_cm)
    if not np.isclose(magnification, 1.0):
        slice_mu = zoom(slice_mu, magnification, order=1)

    # Effective attenuation scaling with kVp (higher kVp -> lower μ)
    mu_eff = slice_mu * (kVp_ref / float(kVp))

    # Rotate slice by projection angle (parallel beam)
    mu_rot = rotate(
        mu_eff,
        angle_deg,
        reshape=False,  # keep same output size as input
        order=1,
        mode="constant",
        cval=0.0,
    )

    # Compute line integrals along beam direction (columns)
    voxel_size_cm = voxel_size_mm * 0.1  # 1 mm = 0.1 cm
    # Sum over rows (axis 0) to get line integrals for each detector column
    L = np.sum(mu_rot, axis=0) * voxel_size_cm  # shape (Nx_rot,)

    # Beer–Lambert law: transmitted intensity (relative)
    transmission_1d = np.exp(-L)  # between 0 and 1

    # Expand to 2D detector (same value for each row)
    proj = np.tile(transmission_1d, (mu_rot.shape[0], 1))

    # Distance-dependent photon flux scaling (inverse square law)
    flux_factor = (distance_ref_cm / float(distance_cm)) ** 2
    mean_intensity = proj * flux_factor  # still roughly in [0, 1]

    # Add noise via a simple Poisson photon counting model
    if noise_level <= 0.0:
        image = mean_intensity
    else:
        # Lower effective photon counts for higher noise_level
        noise_level_clipped = np.clip(noise_level, 0.0, 0.99)
        effective_I0 = max(I0 * (1.0 - noise_level_clipped), 100.0)
        expected_counts = mean_intensity * effective_I0
        noisy_counts = np.random.poisson(expected_counts)
        image = noisy_counts / effective_I0

    # Apply Gaussian blur to mimic system PSF
    if blur_sigma > 0.0:
        image = gaussian_filter(image, sigma=blur_sigma)

    # Clip to [0, 1]
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)
