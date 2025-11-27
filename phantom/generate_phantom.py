"""
Phantom generation for a simplified leg model:
two concentric cylinders (bone inside soft tissue)
aligned with the z-axis.

The phantom is a 3D volume of linear attenuation coefficients μ
[in arbitrary consistent units] representing a μ-map.

Author: (your name)
"""

import numpy as np


def generate_leg_phantom(
    Nz: int = 128,
    Ny: int = 256,
    Nx: int = 256,
    R_tissue_mm: float = 80.0,
    R_bone_mm: float = 40.0,
    mu_tissue: float = 0.25,
    mu_bone: float = 0.6,
    voxel_size_mm: float = 1.0,
) -> np.ndarray:
    """
    Generate a 3D leg phantom with concentric cylinders.

    The cylindrical leg is aligned with the z-axis.
    In cross-section (y,x), the inner circle is bone, outer ring is soft tissue.
    Outside the outer radius is air (μ = 0).

    Parameters
    ----------
    Nz, Ny, Nx : int
        Volume dimensions in voxels (z, y, x).
    R_tissue_mm : float
        Radius of outer soft-tissue cylinder in mm.
    R_bone_mm : float
        Radius of inner bone cylinder in mm.
    mu_tissue : float
        Linear attenuation coefficient for soft tissue.
    mu_bone : float
        Linear attenuation coefficient for bone.
    voxel_size_mm : float
        Isotropic voxel size (dx = dy = dz) in mm.

    Returns
    -------
    phantom : np.ndarray
        3D array (Nz, Ny, Nx) with μ values.
    """
    # Coordinate system (centered around the middle of the FOV).
    y = np.arange(Ny) - Ny / 2 + 0.5
    x = np.arange(Nx) - Nx / 2 + 0.5
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Physical radius in mm.
    R_mm = np.sqrt(X**2 + Y**2) * voxel_size_mm

    phantom = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    tissue_mask = R_mm <= R_tissue_mm
    bone_mask = R_mm <= R_bone_mm

    # Fill with tissue, then overwrite bone region.
    phantom[:, tissue_mask] = mu_tissue
    phantom[:, bone_mask] = mu_bone

    return phantom


def create_default_leg_phantom() -> np.ndarray:
    """
    Convenience function that returns a default leg phantom
    with reasonable parameters.
    """
    return generate_leg_phantom()


if __name__ == "__main__":
    # Quick visual check when run as a script.
    import matplotlib.pyplot as plt

    ph = create_default_leg_phantom()
    Nz, Ny, Nx = ph.shape
    mu_slice = ph[Nz // 2]

    plt.figure(figsize=(4, 4))
    plt.imshow(mu_slice, cmap="viridis", origin="lower")
    plt.colorbar(label="μ (a.u.)")
    plt.title("Central slice of leg phantom μ-map")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
