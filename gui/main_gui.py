"""
Tkinter GUI for Virtual X-Rays of a Simplified Leg Phantom.

Panels:
1. Phantom generation (parameters + μ-map visualization).
2. Acquisition parameters (angle, kVp, distance, noise, blur).
3. Image quality analysis (reference vs test, metrics, difference image).

Author: (your name)
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from phantom.generate_phantom import generate_leg_phantom, create_default_leg_phantom
from simulation.xray_projection import simulate_projection
from metrics.metrics import (
    compute_mse,
    compute_ssim,
    compute_gradient_error,
    compute_gradient_entropy,
)


# Reference parameters (fixed high-quality baseline)
REF_KVP = 70.0
REF_DISTANCE_CM = 100.0
REF_NOISE = 0.0
REF_BLUR = 0.0
REF_I0 = 1.0


class XRaySimulatorGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Virtual X-Rays of a Simplified Leg Phantom")
        self.geometry("1100x700")

        # State
        self.phantom = None
        self.mu_slice = None
        self.current_projection = None
        self.reference_projection = None

        self._build_widgets()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def _build_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.frame_phantom = ttk.Frame(notebook)
        self.frame_acquisition = ttk.Frame(notebook)
        self.frame_analysis = ttk.Frame(notebook)

        notebook.add(self.frame_phantom, text="1. Phantom Generation")
        notebook.add(self.frame_acquisition, text="2. Acquisition")
        notebook.add(self.frame_analysis, text="3. Analysis")

        self._build_phantom_panel()
        self._build_acquisition_panel()
        self._build_analysis_panel()

    # ------------------------------------------------------------------
    # Panel 1: Phantom Generation
    # ------------------------------------------------------------------
    def _build_phantom_panel(self):
        frame = self.frame_phantom

        param_frame = ttk.Labelframe(frame, text="Phantom parameters")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Default values
        self.Nz_var = tk.IntVar(value=128)
        self.Ny_var = tk.IntVar(value=256)
        self.Nx_var = tk.IntVar(value=256)
        self.R_tissue_var = tk.DoubleVar(value=80.0)
        self.R_bone_var = tk.DoubleVar(value=40.0)
        self.mu_tissue_var = tk.DoubleVar(value=0.25)
        self.mu_bone_var = tk.DoubleVar(value=0.6)
        self.voxel_size_var = tk.DoubleVar(value=1.0)

        def add_row(parent, label, var, row):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=var, width=10).grid(
                row=row, column=1, sticky="w", padx=5, pady=2
            )

        add_row(param_frame, "Nz (slices)", self.Nz_var, 0)
        add_row(param_frame, "Ny (pixels)", self.Ny_var, 1)
        add_row(param_frame, "Nx (pixels)", self.Nx_var, 2)
        add_row(param_frame, "R_tissue (mm)", self.R_tissue_var, 3)
        add_row(param_frame, "R_bone (mm)", self.R_bone_var, 4)
        add_row(param_frame, "μ_tissue", self.mu_tissue_var, 5)
        add_row(param_frame, "μ_bone", self.mu_bone_var, 6)
        add_row(param_frame, "Voxel size (mm)", self.voxel_size_var, 7)

        ttk.Button(
            param_frame,
            text="Generate Phantom",
            command=self.on_generate_phantom,
        ).grid(row=8, column=0, columnspan=2, pady=10)

        # Phantom visualization
        vis_frame = ttk.Labelframe(frame, text="Central μ-map slice")
        vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.phantom_fig = Figure(figsize=(4, 4))
        self.phantom_ax = self.phantom_fig.add_subplot(111)
        self.phantom_ax.set_title("No phantom yet")
        self.phantom_ax.axis("off")

        self.phantom_canvas = FigureCanvasTkAgg(self.phantom_fig, master=vis_frame)
        self.phantom_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_generate_phantom(self):
        try:
            Nz = int(self.Nz_var.get())
            Ny = int(self.Ny_var.get())
            Nx = int(self.Nx_var.get())
            R_tissue = float(self.R_tissue_var.get())
            R_bone = float(self.R_bone_var.get())
            mu_tissue = float(self.mu_tissue_var.get())
            mu_bone = float(self.mu_bone_var.get())
            voxel_size = float(self.voxel_size_var.get())
        except Exception:
            messagebox.showerror("Error", "Invalid phantom parameters.")
            return

        if R_bone > R_tissue:
            messagebox.showerror("Error", "Bone radius must be <= tissue radius.")
            return

        self.phantom = generate_leg_phantom(
            Nz=Nz,
            Ny=Ny,
            Nx=Nx,
            R_tissue_mm=R_tissue,
            R_bone_mm=R_bone,
            mu_tissue=mu_tissue,
            mu_bone=mu_bone,
            voxel_size_mm=voxel_size,
        )
        self.mu_slice = self.phantom[Nz // 2]

        # Update plot
        self.phantom_ax.clear()
        im = self.phantom_ax.imshow(
            self.mu_slice, cmap="viridis", origin="lower"
        )
        self.phantom_ax.set_title("Central μ-map slice")
        self.phantom_ax.axis("off")
        self.phantom_fig.colorbar(
            im, ax=self.phantom_ax, fraction=0.046, pad=0.04
        )
        self.phantom_canvas.draw()

        messagebox.showinfo("Phantom", "Phantom generated successfully.")

    # ------------------------------------------------------------------
    # Panel 2: Acquisition
    # ------------------------------------------------------------------
    def _build_acquisition_panel(self):
        frame = self.frame_acquisition

        controls = ttk.Labelframe(frame, text="Acquisition parameters")
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Variables
        self.angle_var = tk.DoubleVar(value=0.0)
        self.kvp_var = tk.DoubleVar(value=70.0)
        self.distance_var = tk.DoubleVar(value=100.0)
        self.noise_var = tk.DoubleVar(value=0.0)
        self.blur_var = tk.DoubleVar(value=0.0)

        def add_scale(parent, label, var, from_, to_, row, resolution=1.0):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            scale = tk.Scale(
                parent,
                variable=var,
                from_=from_,
                to=to_,
                orient=tk.HORIZONTAL,
                resolution=resolution,
                length=200,
            )
            scale.grid(row=row, column=1, padx=5, pady=5, sticky="we")
            return scale

        add_scale(controls, "Angle (deg)", self.angle_var, 0, 180, row=0, resolution=1.0)
        add_scale(controls, "kVp", self.kvp_var, 40, 120, row=1, resolution=1.0)
        add_scale(
            controls,
            "Distance (cm)",
            self.distance_var,
            80,
            140,
            row=2,
            resolution=1.0,
        )
        add_scale(
            controls,
            "Noise level",
            self.noise_var,
            0.0,
            1.0,
            row=3,
            resolution=0.05,
        )
        add_scale(
            controls,
            "Blur σ (pixels)",
            self.blur_var,
            0.0,
            3.0,
            row=4,
            resolution=0.1,
        )

        ttk.Button(
            controls,
            text="Simulate X-Ray",
            command=self.on_simulate,
        ).grid(row=5, column=0, columnspan=2, pady=10)

        ttk.Label(
            controls,
            text=(
                "Reference image:\n"
                f"kVp={REF_KVP}, distance={REF_DISTANCE_CM} cm,\n"
                "noise=0, blur=0 (same angle as test)."
            ),
            foreground="gray",
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=5)

        # Projection visualization (current test)
        vis_frame = ttk.Labelframe(frame, text="Current projection (test)")
        vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.proj_fig = Figure(figsize=(4, 4))
        self.proj_ax = self.proj_fig.add_subplot(111)
        self.proj_ax.set_title("No projection yet")
        self.proj_ax.axis("off")

        self.proj_canvas = FigureCanvasTkAgg(self.proj_fig, master=vis_frame)
        self.proj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_simulate(self):
        if self.phantom is None:
            messagebox.showerror(
                "Error", "Please generate a phantom first (tab 1)."
            )
            return

        angle = float(self.angle_var.get())
        kvp = float(self.kvp_var.get())
        distance = float(self.distance_var.get())
        noise_level = float(self.noise_var.get())
        blur_sigma = float(self.blur_var.get())

        # Test image
        proj_test, proj_clean, mu_slice, mu_rot, line_int = simulate_projection(
            self.phantom,
            angle_deg=angle,
            kVp=kvp,
            distance_cm=distance,
            I0=1.0,
            noise_level=noise_level,
            blur_sigma=blur_sigma,
        )

        self.current_projection = proj_test
        self.mu_slice = mu_slice  # keep reference

        # Reference image (same angle, fixed high-quality settings)
        proj_ref, _, _, _, _ = simulate_projection(
            self.phantom,
            angle_deg=angle,
            kVp=REF_KVP,
            distance_cm=REF_DISTANCE_CM,
            I0=REF_I0,
            noise_level=REF_NOISE,
            blur_sigma=REF_BLUR,
        )
        self.reference_projection = proj_ref

        # Update test projection display
        self.proj_ax.clear()
        self.proj_ax.imshow(
            self.current_projection, cmap="gray", origin="lower"
        )
        self.proj_ax.set_title("Current test projection")
        self.proj_ax.axis("off")
        self.proj_canvas.draw()

        # Update analysis
        self.update_analysis_panel()

    # ------------------------------------------------------------------
    # Panel 3: Analysis
    # ------------------------------------------------------------------
    def _build_analysis_panel(self):
        frame = self.frame_analysis

        # Metrics display
        metrics_frame = ttk.Labelframe(frame, text="Image quality metrics")
        metrics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.mse_label = ttk.Label(metrics_frame, text="MSE: -")
        self.ssim_label = ttk.Label(metrics_frame, text="SSIM: -")
        self.ge_label = ttk.Label(metrics_frame, text="Gradient Error (GE): -")
        self.gen_ref_label = ttk.Label(metrics_frame, text="gEn (ref): -")
        self.gen_test_label = ttk.Label(metrics_frame, text="gEn (test): -")

        self.mse_label.pack(anchor="w", pady=2)
        self.ssim_label.pack(anchor="w", pady=2)
        self.ge_label.pack(anchor="w", pady=2)
        self.gen_ref_label.pack(anchor="w", pady=2)
        self.gen_test_label.pack(anchor="w", pady=2)

        ttk.Button(
            metrics_frame,
            text="Plot Intensity Profile",
            command=self.on_plot_profile,
        ).pack(pady=10)

        # Images: reference, test, difference
        vis_frame = ttk.Labelframe(frame, text="Images")
        vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.analysis_fig = Figure(figsize=(8, 3))
        self.ax_ref = self.analysis_fig.add_subplot(1, 3, 1)
        self.ax_test = self.analysis_fig.add_subplot(1, 3, 2)
        self.ax_diff = self.analysis_fig.add_subplot(1, 3, 3)

        for ax, title in zip(
            (self.ax_ref, self.ax_test, self.ax_diff),
            ("Reference", "Test", "Abs. difference"),
        ):
            ax.set_title(title)
            ax.axis("off")

        self.analysis_canvas = FigureCanvasTkAgg(
            self.analysis_fig, master=vis_frame
        )
        self.analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_analysis_panel(self):
        if self.reference_projection is None or self.current_projection is None:
            return

        ref = self.reference_projection
        test = self.current_projection

        if ref.shape != test.shape:
            messagebox.showwarning(
                "Warning", "Reference and test images have different shapes."
            )
            return

        # Compute metrics
        mse_val = compute_mse(ref, test)
        ssim_val = compute_ssim(ref, test)
        ge_val = compute_gradient_error(ref, test)
        gen_ref = compute_gradient_entropy(ref)
        gen_test = compute_gradient_entropy(test)

        self.mse_label.config(text=f"MSE: {mse_val:.5f}")
        self.ssim_label.config(text=f"SSIM: {ssim_val:.4f}")
        self.ge_label.config(text=f"Gradient Error (GE): {ge_val:.5f}")
        self.gen_ref_label.config(text=f"gEn (ref): {gen_ref:.4f}")
        self.gen_test_label.config(text=f"gEn (test): {gen_test:.4f}")

        # Update images
        diff = np.abs(ref - test)

        self.ax_ref.clear()
        self.ax_ref.imshow(ref, cmap="gray", origin="lower")
        self.ax_ref.set_title("Reference")
        self.ax_ref.axis("off")

        self.ax_test.clear()
        self.ax_test.imshow(test, cmap="gray", origin="lower")
        self.ax_test.set_title("Test")
        self.ax_test.axis("off")

        self.ax_diff.clear()
        self.ax_diff.imshow(diff, cmap="gray", origin="lower")
        self.ax_diff.set_title("|Ref - Test|")
        self.ax_diff.axis("off")

        self.analysis_canvas.draw()

    def on_plot_profile(self):
        if self.reference_projection is None or self.current_projection is None:
            messagebox.showerror(
                "Error",
                "No reference/test images. Run a simulation first.",
            )
            return

        ref = self.reference_projection
        test = self.current_projection
        center_row = ref.shape[0] // 2
        x = np.arange(ref.shape[1])

        plt.figure(figsize=(5, 3))
        plt.plot(x, ref[center_row, :], label="Reference")
        plt.plot(x, test[center_row, :], label="Test", linestyle="--")
        plt.xlabel("Detector pixel")
        plt.ylabel("Intensity (a.u.)")
        plt.title("Central horizontal intensity profile")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    app = XRaySimulatorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
