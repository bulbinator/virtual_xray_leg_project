import numpy as np

def generate_leg_phantom(
    Nz: int = 128,
    Ny: int = 256,
    Nx: int = 256,
    voxel_size_mm: float = 1.0,
    R_tissue_mm: float = 80.0,
    R_bone_mm: float = 40.0,
    mu_tissue: float = 0.25,
    mu_bone: float = 0.6,
) -> np.ndarray:
    """
    Generate a simplified 3D leg phantom made of two concentric cylinders.

    The cylinder axis is along the z-direction. Each voxel stores a linear
    attenuation coefficient μ (in 1/cm).

    Parameters
    ----------
    Nz, Ny, Nx : int
        Number of voxels along z, y, x.
    voxel_size_mm : float
        Isotropic voxel size in mm.
    R_tissue_mm : float
        Outer radius of the soft-tissue cylinder in mm.
    R_bone_mm : float
        Inner radius of the bone cylinder in mm (must be <= R_tissue_mm).
    mu_tissue : float
        Linear attenuation coefficient for soft tissue (1/cm).
    mu_bone : float
        Linear attenuation coefficient for bone (1/cm).

    Returns
    -------
    phantom : np.ndarray
        3D array of shape (Nz, Ny, Nx) containing μ values.
    """
    if R_bone_mm > R_tissue_mm:
        raise ValueError("R_bone_mm must be smaller than or equal to R_tissue_mm")

    # Coordinate grids in mm (centered)
    yy = (np.arange(Ny) - (Ny - 1) / 2.0) * voxel_size_mm
    xx = (np.arange(Nx) - (Nx - 1) / 2.0) * voxel_size_mm
    X, Y = np.meshgrid(xx, yy)
    r_mm = np.sqrt(X**2 + Y**2)

    # Initialize slice with soft-tissue μ
    slice_mu = np.full((Ny, Nx), mu_tissue, dtype=np.float32)

    # Set bone region
    slice_mu[r_mm <= R_bone_mm] = mu_bone
    # Outside tissue cylinder is air (μ ~ 0)
    slice_mu[r_mm > R_tissue_mm] = 0.0

    # Extrude along z
    phantom = np.repeat(slice_mu[None, ...], Nz, axis=0)
    return phantom
