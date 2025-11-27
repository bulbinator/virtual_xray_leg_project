import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from phantom import generate_leg_phantom
from xray_simulator import simulate_projection
from metrics.metrics import compute_all_metrics, _resize_to_match  # type: ignore


class XRaySimulatorGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Virtual X-Rays of a Simplified Leg Phantom")

        # State
        self.phantom: Optional[np.ndarray] = None
        self.reference_image: Optional[np.ndarray] = None
        self.test_image: Optional[np.ndarray] = None

        # Layout: main frames
        self._build_phantom_frame()
        self._build_acquisition_frame()
        self._build_metrics_frame()
        self._build_plot_area()

    # ---------------- Phantom frame ----------------
    def _build_phantom_frame(self) -> None:
        frame = ttk.LabelFrame(self.master, text="1. Phantom Generation")
        frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Size entries
        ttk.Label(frame, text="Nz").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(frame, text="Ny").grid(row=0, column=2, sticky=tk.W)
        ttk.Label(frame, text="Nx").grid(row=0, column=4, sticky=tk.W)

        self.Nz_var = tk.IntVar(value=128)
        self.Ny_var = tk.IntVar(value=256)
        self.Nx_var = tk.IntVar(value=256)

        ttk.Entry(frame, textvariable=self.Nz_var, width=6).grid(row=0, column=1, padx=2)
        ttk.Entry(frame, textvariable=self.Ny_var, width=6).grid(row=0, column=3, padx=2)
        ttk.Entry(frame, textvariable=self.Nx_var, width=6).grid(row=0, column=5, padx=2)

        # Radii and μ values
        ttk.Label(frame, text="R_tissue [mm]").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(frame, text="R_bone [mm]").grid(row=1, column=2, sticky=tk.W)
        ttk.Label(frame, text="μ_tissue [1/cm]").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(frame, text="μ_bone [1/cm]").grid(row=2, column=2, sticky=tk.W)

        self.R_tissue_var = tk.DoubleVar(value=80.0)
        self.R_bone_var = tk.DoubleVar(value=40.0)
        self.mu_tissue_var = tk.DoubleVar(value=0.25)
        self.mu_bone_var = tk.DoubleVar(value=0.6)

        ttk.Entry(frame, textvariable=self.R_tissue_var, width=8).grid(row=1, column=1, padx=2)
        ttk.Entry(frame, textvariable=self.R_bone_var, width=8).grid(row=1, column=3, padx=2)
        ttk.Entry(frame, textvariable=self.mu_tissue_var, width=8).grid(row=2, column=1, padx=2)
        ttk.Entry(frame, textvariable=self.mu_bone_var, width=8).grid(row=2, column=3, padx=2)

        ttk.Button(frame, text="Generate Phantom", command=self.generate_phantom).grid(
            row=1, column=5, rowspan=2, padx=10, pady=2
        )

    # ---------------- Acquisition frame ----------------
    def _build_acquisition_frame(self) -> None:
        frame = ttk.LabelFrame(self.master, text="2. Acquisition Parameters")
        frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Angle
        ttk.Label(frame, text="Angle [deg]").grid(row=0, column=0, sticky=tk.W)
        self.angle_var = tk.DoubleVar(value=0.0)
        ttk.Scale(frame, from_=0.0, to=180.0, variable=self.angle_var, orient=tk.HORIZONTAL).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=4
        )

        # kVp
        ttk.Label(frame, text="kVp").grid(row=1, column=0, sticky=tk.W)
        self.kvp_var = tk.DoubleVar(value=70.0)
        ttk.Scale(frame, from_=40.0, to=120.0, variable=self.kvp_var, orient=tk.HORIZONTAL).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=4
        )

        # Distance
        ttk.Label(frame, text="Distance [cm]").grid(row=2, column=0, sticky=tk.W)
        self.distance_var = tk.DoubleVar(value=100.0)
        ttk.Scale(frame, from_=60.0, to=140.0, variable=self.distance_var, orient=tk.HORIZONTAL).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=4
        )

        # Noise level
        ttk.Label(frame, text="Noise level").grid(row=3, column=0, sticky=tk.W)
        self.noise_var = tk.DoubleVar(value=0.3)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.noise_var, orient=tk.HORIZONTAL).grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=4
        )

        # Blur sigma
        ttk.Label(frame, text="Blur σ [px]").grid(row=4, column=0, sticky=tk.W)
        self.blur_var = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.0, to=3.0, variable=self.blur_var, orient=tk.HORIZONTAL).grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=4
        )

        # Simulate button
        ttk.Button(frame, text="Simulate X-Ray", command=self.simulate_xray).grid(
            row=0, column=4, rowspan=3, padx=10, pady=2
        )

    # ---------------- Metrics frame ----------------
    def _build_metrics_frame(self) -> None:
        frame = ttk.LabelFrame(self.master, text="3. Image Quality Metrics")
        frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(frame, text="MSE:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(frame, text="SSIM:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(frame, text="Gradient Error:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(frame, text="Gradient Entropy (ref):").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(frame, text="Gradient Entropy (test):").grid(row=4, column=0, sticky=tk.W)

        self.mse_var = tk.StringVar(value="N/A")
        self.ssim_var = tk.StringVar(value="N/A")
        self.ge_var = tk.StringVar(value="N/A")
        self.gen_ref_var = tk.StringVar(value="N/A")
        self.gen_test_var = tk.StringVar(value="N/A")

        ttk.Label(frame, textvariable=self.mse_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(frame, textvariable=self.ssim_var).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(frame, textvariable=self.ge_var).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(frame, textvariable=self.gen_ref_var).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(frame, textvariable=self.gen_test_var).grid(row=4, column=1, sticky=tk.W)

    # ---------------- Plot area ----------------
    def _build_plot_area(self) -> None:
        frame = ttk.LabelFrame(self.master, text="Phantom and Projections")
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(8, 6))
        self.ax_phantom = self.fig.add_subplot(2, 2, 1)
        self.ax_ref = self.fig.add_subplot(2, 2, 2)
        self.ax_test = self.fig.add_subplot(2, 2, 3)
        self.ax_diff = self.fig.add_subplot(2, 2, 4)

        for ax in (self.ax_phantom, self.ax_ref, self.ax_test, self.ax_diff):
            ax.axis("off")

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------------- Actions ----------------
    def generate_phantom(self) -> None:
        Nz = self.Nz_var.get()
        Ny = self.Ny_var.get()
        Nx = self.Nx_var.get()
        R_tissue = self.R_tissue_var.get()
        R_bone = self.R_bone_var.get()
        mu_tissue = self.mu_tissue_var.get()
        mu_bone = self.mu_bone_var.get()

        self.phantom = generate_leg_phantom(
            Nz=Nz,
            Ny=Ny,
            Nx=Nx,
            R_tissue_mm=R_tissue,
            R_bone_mm=R_bone,
            mu_tissue=mu_tissue,
            mu_bone=mu_bone,
        )

        # Show central slice of μ-map
        Nz_actual = self.phantom.shape[0]
        phantom_slice = self.phantom[Nz_actual // 2, :, :]

        self.ax_phantom.clear()
        im = self.ax_phantom.imshow(phantom_slice, cmap="gray")
        self.ax_phantom.set_title("Phantom μ-map (central slice)")
        self.ax_phantom.axis("off")
        # Manage colorbar: create once and update
        if not hasattr(self, "_phantom_colorbar"):
            self._phantom_colorbar = self.fig.colorbar(
                im, ax=self.ax_phantom, fraction=0.046, pad=0.04
            )
        else:
            self._phantom_colorbar.update_normal(im)

        self.fig.tight_layout()
        self.canvas.draw()

    def simulate_xray(self) -> None:
        if self.phantom is None:
            messagebox.showerror("Error", "Please generate the phantom first.")
            return

        angle = float(self.angle_var.get())
        kvp = float(self.kvp_var.get())
        distance = float(self.distance_var.get())
        noise = float(self.noise_var.get())
        blur = float(self.blur_var.get())

        # Reference image: standard settings, no noise/blur
        self.reference_image = simulate_projection(
            self.phantom,
            angle_deg=angle,
            kVp=70.0,
            distance_cm=100.0,
            noise_level=0.0,
            blur_sigma=0.0,
        )

        # Test image: user-selected parameters
        self.test_image = simulate_projection(
            self.phantom,
            angle_deg=angle,
            kVp=kvp,
            distance_cm=distance,
            noise_level=noise,
            blur_sigma=blur,
        )

        # Resize test to match reference for difference image
        test_resized = _resize_to_match(self.reference_image, self.test_image)
        diff_image = np.abs(self.reference_image - test_resized)

        # Update plots
        for ax in (self.ax_ref, self.ax_test, self.ax_diff):
            ax.clear()
            ax.axis("off")

        if self.reference_image is not None:
            self.ax_ref.imshow(self.reference_image, cmap="gray")
            self.ax_ref.set_title("Reference projection")

        if self.test_image is not None:
            self.ax_test.imshow(self.test_image, cmap="gray")
            self.ax_test.set_title("Test projection")

        self.ax_diff.imshow(diff_image, cmap="hot")
        self.ax_diff.set_title("|Ref - Test|")

        self.fig.tight_layout()
        self.canvas.draw()

        # Compute and display metrics
        metrics = compute_all_metrics(self.reference_image, self.test_image)
        self.mse_var.set(f"{metrics['MSE']:.4f}")
        self.ssim_var.set(f"{metrics['SSIM']:.4f}")
        self.ge_var.set(f"{metrics['GradientError']:.4f}")
        self.gen_ref_var.set(f"{metrics['GradientEntropy_ref']:.4f}")
        self.gen_test_var.set(f"{metrics['GradientEntropy_test']:.4f}")


def main() -> None:
    root = tk.Tk()
    app = XRaySimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
