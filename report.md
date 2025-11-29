# Virtual X-Rays of a Simplified Leg Phantom

## 1. Members

- **Name:** _[Your Name Here]_  
- **Student ID:** _[ID]_  
- **Email:** _[email@domain]_  
- **Program:** _[BSc/MSc/PhD in ...]_  

---

## 2. Introduction

Conventional X-ray imaging is widely used for musculoskeletal assessment, in particular
for detecting fractures and evaluating bone alignment in the legs. Working with real
X-ray systems, however, is costly, involves ionizing radiation, and is not always
convenient for teaching and prototyping.

In this project we build a **computational simulator** of a simplified X-ray imaging
setup. The imaged object is a **virtual leg phantom** modeled as two concentric
cylinders: an outer soft-tissue region and an inner bone region with higher attenuation.
The phantom encodes a ground-truth 3D distribution of linear attenuation coefficients μ.

Using this phantom as the ground truth, we simulate 2D X-ray projection images while
varying acquisition parameters such as beam energy (kVp), source–detector distance,
projection angle, noise level and blur. We then evaluate image quality using metrics
from the *Evaluation Metrics in Medical Imaging* lecture.

### 2.1 Aims

**Aim 1.** Implement a simplified cylindrical leg phantom with bone and soft-tissue
regions and configurable geometry and attenuation coefficients.

**Aim 2.** Implement a forward model for X-ray projection imaging based on the
Beer–Lambert law, including the effects of projection angle, beam energy (kVp),
source–detector distance, noise and blur.

**Aim 3.** Develop a Python GUI that allows interactive phantom generation, parameter
adjustment and visualization of simulated X-ray images.

**Aim 4.** Quantitatively evaluate image quality using MSE, SSIM, Gradient Error and
Gradient Entropy, and analyze how these metrics depend on acquisition parameters.

---

## 3. Methods

### 3.1 Phantom design (Aim 1)

The leg phantom is represented as a 3D array:

- **Size:** \(N_z \times N_y \times N_x\) voxels (default: 128×256×256).
- **Voxel size:** isotropic, \(d_x = d_y = d_z = 1\) mm.
- **Geometry:** two concentric cylinders aligned with the z-axis:
  - Outer cylinder (soft tissue) of radius \(R_\text{tissue}\).
  - Inner cylinder (bone) of radius \(R_\text{bone}\) (with \(R_\text{bone} < R_\text{tissue}\)).

Each voxel stores a linear attenuation coefficient μ (in 1/cm):

- \( \mu_\text{tissue} \approx 0.25 \ \text{cm}^{-1} \)
- \( \mu_\text{bone} \approx 0.60 \ \text{cm}^{-1} \)
- Voxels outside \(R_\text{tissue}\) are treated as air with μ ≈ 0.

We generate the phantom by creating a 2D μ-map in the central x–y plane and extruding
it along z. For each voxel (x, y) we compute the radius
\( r = \sqrt{x^2 + y^2} \) in mm; if \(r \le R_\text{bone}\) we assign μ = μ_bone,
else if \(r \le R_\text{tissue}\) we assign μ = μ_tissue, otherwise μ = 0.

This is implemented in `generate_leg_phantom` (phantom.py).

### 3.2 X-ray forward model (Aim 2)

We model X-ray imaging with a simplified **parallel-beam** geometry. The key equation is
the Beer–Lambert law:

\[
I = I_0 \exp\left(-\int_{\text{ray}} \mu(s) \, ds\right)
\]

On a discrete grid this becomes:

\[
I = I_0 \exp\left(-\sum_k \mu_k \, \Delta s\right)
\]

where μ_k are sample values along each ray and \(\Delta s\) is the voxel size in cm.

#### 3.2.1 Central slice and parallel-beam projection

To reduce complexity, we simulate projections of a single central slice:

1. Extract the central slice of the 3D phantom:
   \[
   \mu_\text{slice}(y, x) = \text{phantom}[z = N_z/2, y, x].
   \]

2. **Distance-dependent magnification.** We approximate geometric magnification by
   scaling the slice according to the source–detector distance \(D\):

   - Reference distance: \(D_\text{ref} = 100\) cm.
   - Magnification factor:
     \[
     M(D) = \frac{D_\text{ref}}{D}
     \]
     so that larger D → less magnification.

   The slice is resized using bilinear interpolation (scipy.ndimage.zoom).

3. **Effective attenuation vs kVp.** Instead of a full spectrum model, we use a simple
   effective-μ scaling:

   - Reference kVp: \(kVp_\text{ref} = 70\) kVp.
   - Effective μ at given kVp:
     \[
     \mu_\text{eff} = \mu \cdot \frac{kVp_\text{ref}}{kVp}
     \]
     so higher kVp leads to lower effective μ and thus higher transmission.

4. **Rotation by projection angle.** We apply a 2D rotation of the μ-map around its
   center using `scipy.ndimage.rotate` with `reshape=False`, which corresponds to a
   parallel-beam projection at angle θ in the x–y plane.

5. **Line integrals and intensity.** We sum μ_eff along the beam direction (columns):

   \[
   L(x_\text{det}) = \sum_y \mu_\text{rot}(y, x_\text{det}) \, \Delta s
                   = \sum_y \mu_\text{rot}(y, x_\text{det}) \, d_\text{voxel,cm}
   \]

   with \( d_\text{voxel,cm} = 0.1 \) cm (for 1 mm voxels). For each detector column:

   \[
   T(x_\text{det}) = \exp(-L(x_\text{det}))
   \]

   We then tile this 1D transmission profile along the detector rows to obtain a 2D
   projection image.

#### 3.2.2 Distance, flux, noise and blur

We include a simple model of distance-dependent photon flux and image degradation:

- **Inverse-square law:** photon fluence at the detector scales as
  \[
  \text{flux}(D) \propto \left(\frac{D_\text{ref}}{D}\right)^2.
  \]
  The mean intensity image is:
  \[
  I_\text{mean}(y,x) = T(y,x) \cdot \left(\frac{D_\text{ref}}{D}\right)^2.
  \]

- **Photon counting noise (Poisson model):**
  - Let \(I_0\) be the baseline number of photons per pixel at \(D_\text{ref}\).
  - For a given noise level \(\alpha \in [0,1]\) we define an effective photon budget
    \[
    I_0^\text{eff} = \max\big(I_0 (1 - \alpha), 100\big),
    \]
    so higher noise level → fewer photons.
  - Expected photon counts:
    \[
    N_\text{exp}(y,x) = I_\text{mean}(y,x) \cdot I_0^\text{eff}.
    \]
  - Simulated noisy counts:
    \[
    N_\text{noisy}(y,x) \sim \text{Poisson}\big(N_\text{exp}(y,x)\big),
    \]
    and the normalized image is
    \[
    I_\text{noisy}(y,x) = \frac{N_\text{noisy}(y,x)}{I_0^\text{eff}}.
    \]

- **Blur (system PSF):** we apply a Gaussian filter with standard deviation σ in pixels
  using `scipy.ndimage.gaussian_filter`:
  \[
  I_\text{blur} = G_\sigma * I_\text{noisy}.
  \]

Finally the image is clipped to [0, 1]. All of this is implemented in
`simulate_projection` (xray_simulator.py).

### 3.3 Image quality metrics (Aim 4)

We use four metrics to quantify image quality, implemented in `metrics/metrics.py`.

#### 3.3.1 Reference images

For each projection angle θ, we define a high-quality reference image:

- \(kVp = 70\) kVp
- Distance \(D = 100\) cm
- Noise level = 0.0
- Blur σ = 0.0

This gives a noise-free, unblurred projection dominated by the phantom geometry and
Beer–Lambert attenuation.

#### 3.3.2 Mean Squared Error (MSE)

For reference image \(R\) and test image \(T\) (resampled to the same size) we compute:

\[
\text{MSE}(R, T) = \frac{1}{N} \sum_{i=1}^{N} (R_i - T_i)^2.
\]

Lower MSE indicates that the test image is closer to the reference.

#### 3.3.3 Structural Similarity Index (SSIM)

We use either the `skimage.metrics.structural_similarity` implementation or, if
`scikit-image` is not installed, a simple global SSIM approximation based on means and
variances of R and T. SSIM is defined in terms of luminance, contrast and structure
similarity, and typically varies between 0 (no similarity) and 1 (identical images).

#### 3.3.4 Gradient Error (GE)

We compute gradient magnitudes using Sobel filters:

\[
G(I) = \sqrt{(\partial_x I)^2 + (\partial_y I)^2}.
\]

The gradient error is defined as the mean absolute difference between gradient
magnitudes:

\[
\text{GE}(R, T) = \frac{1}{N} \sum_{i=1}^{N} |G(R)_i - G(T)_i|.
\]

GE is sensitive to edge preservation and blurring.

#### 3.3.5 Gradient Entropy (gEn)

Gradient entropy is a no-reference sharpness measure:

1. Compute gradient magnitudes \(G(I)\) for an image.
2. Build a histogram of G with \(K\) bins; let \(p_k\) be the normalized bin counts.
3. Compute
   \[
   \text{gEn}(I) = -\sum_{k=1}^{K} p_k \log_2 p_k.
   \]

Higher gEn indicates richer edge content and sharper images.

We compute gEn for both reference and test images. All metrics are wrapped in
`compute_all_metrics`, which returns a dictionary of metric values.

### 3.4 GUI design (Aim 3)

The GUI is implemented with Tkinter in `gui.py` and organized into three main panels:

1. **Phantom Generation Panel**
   - Inputs: Nz, Ny, Nx, \(R_\text{tissue}\), \(R_\text{bone}\), μ_tissue, μ_bone.
   - Button: **Generate Phantom**
   - Output: central μ-slice displayed in the top-left subplot.

2. **Acquisition Parameters Panel**
   - Angle slider (0–180°).
   - kVp slider (40–120 kVp).
   - Distance slider (60–140 cm).
   - Noise-level slider (0–1).
   - Blur σ slider (0–3 pixels).
   - Button: **Simulate X-Ray**.

3. **Image Quality Metrics Panel**
   - Displays numerical values of MSE, SSIM, GE, gEn(ref), gEn(test).

The plotting area (Matplotlib embedded in Tkinter) displays:

- Top-left: phantom μ-map (central slice).
- Top-right: reference projection.
- Bottom-left: test projection.
- Bottom-right: absolute difference |Ref − Test|.

---

## 4. Results and Discussion

### 4.1 Qualitative results

Representative examples of simulated X-ray projections are shown for varying kVp,
distance, noise and blur:

- At **low kVp** (e.g. 50 kVp), bone appears much darker than soft tissue with high
  contrast but overall lower transmission (darker image).
- At **high kVp** (e.g. 90 kVp), bone–tissue contrast visibly decreases and the image
  becomes more uniformly bright.
- Increasing **distance** (with the inverse-square scaling) reduces overall intensity
  and increases noise for a fixed exposure budget.
- Increasing **blur σ** visibly smooths bone edges and softens the transition between
  bone and tissue regions.

The difference images |Ref − Test| highlight parameter-induced changes: strong contrast
differences at low/high kVp and edge softening for larger blur.

### 4.2 Quantitative trends

Using `analysis/run_experiments.py`, we swept:

- Angles: 0°, 30°, 60°, 90°
- kVp: 50, 60, 70, 80, 90
- Distances: 80, 100, 120 cm
- Noise levels: 0.0, 0.3, 0.6
- Blur σ: 0.0, 1.0, 2.0

For each combination we computed the metrics versus a reference at the same angle
(\(kVp = 70\), \(D = 100\) cm, no noise, no blur).

#### 4.2.1 Effect of kVp

- **MSE:** increases as kVp moves away from the reference (70 kVp) in either direction,
  with the largest deviation at the lowest kVp (50) due to stronger attenuation and
  darker images.
- **SSIM:** highest near 70 kVp and decreases for both lower and higher kVp as the
  bone–tissue contrast and overall brightness diverge from the reference.
- **gEn:** tends to decrease at high kVp because reduced contrast diminishes gradient
  magnitudes, leading to a more concentrated gradient histogram.

These trends align with the expectation that mid-range kVp achieves a balance between
contrast and penetration for this simple model.

#### 4.2.2 Effect of distance and noise

- **Distance:** increasing distance (with inverse-square scaling) reduces effective
  photon counts, producing noisier images. For fixed noise-level slider setting:
  - MSE increases with distance.
  - SSIM decreases with distance.
- **Noise level:** directly controls the effective photon budget:
  - Higher noise levels increase MSE and reduce SSIM.
  - Gradient Error grows because noise perturbs edge gradients and introduces
    high-frequency fluctuations.

#### 4.2.3 Effect of blur

- **Blur σ:** primarily affects sharpness metrics:
  - Gradient Error increases with blur since edges become less steep in the test image
    compared to the sharp reference.
  - Gradient Entropy decreases with larger blur, as gradients become weaker and more
    concentrated around small values.

Overall, the metrics behave consistently with intuition from imaging physics and
image processing: better-conserved structure and edges correspond to lower MSE, higher
SSIM, lower GE and higher gEn.

### 4.3 Dependence on projection angle

Because the phantom is cylindrically symmetric, projections at different angles are
theoretically equivalent. In practice, minor differences arise due to the discrete
grid, interpolation during rotation and noise. Metrics remain very similar across
angles, which is consistent with the rotational symmetry of the phantom.

---

## 5. Conclusions

We implemented a complete pipeline for simulating X-ray projections of a simplified
leg phantom and quantitatively evaluating image quality:

1. A 3D leg phantom with soft-tissue and bone regions provides a simple, controllable
   ground truth μ-map.
2. A parallel-beam forward model based on Beer–Lambert law, with simplified kVp and
   distance models, generates realistic-looking X-ray projections.
3. A Tkinter-based GUI allows interactive exploration of acquisition parameters and
   immediate feedback through images and metrics.
4. Image quality metrics (MSE, SSIM, Gradient Error, Gradient Entropy) capture expected
   trends with respect to kVp, distance, noise and blur.

### 5.1 Limitations

- The use of a **parallel-beam** model ignores cone-beam geometry and realistic
  magnification effects.
- The beam energy model is a simple scaling of μ and does not represent a full X-ray
  spectrum or energy-dependent detector response.
- The phantom is highly simplified (cylindrical, homogeneous regions) and does not
  include anatomical details such as marrow, muscle, fat or complex bone structures.
- We only simulate 2D projections of a single central slice rather than full 3D cone-beam
  projections.

### 5.2 Future work

Possible extensions include:

- More realistic anatomical phantoms (e.g., multiple tissue types, irregular bone
  shapes, fractures).
- Polychromatic X-ray spectrum models and energy-dependent μ values.
- True cone-beam geometry with divergent rays and realistic source–detector distances.
- Addition of scatter models and detector-specific noise characteristics.
- Integration of reconstruction algorithms (e.g. filtered backprojection) to simulate
  CT from the same phantom.

---

## 6. Literature

Example references (replace or extend as needed):

1. J. T. Bushberg et al., *The Essential Physics of Medical Imaging*, 3rd ed.,
   Lippincott Williams & Wilkins, 2011.
2. Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment:
   From error visibility to structural similarity," *IEEE Transactions on Image
   Processing*, vol. 13, no. 4, pp. 600–612, 2004.
3. Lecture notes: *Evaluation Metrics in Medical Imaging*, course material.
4. Lecture notes: *X-ray Imaging Physics*, course material.

---

## 7. Code overview and usage

### 7.1 Main files

- `phantom.py`  
  Generates the leg phantom (`generate_leg_phantom`).

- `xray_simulator.py`  
  Computes X-ray projections (`simulate_projection`).

- `metrics/metrics.py`  
  Implements MSE, SSIM, Gradient Error, Gradient Entropy and a convenience
  `compute_all_metrics` function.

- `gui.py`  
  Tkinter GUI for interactive simulations.

- `analysis/run_experiments.py`  
  Parameter sweep script; outputs images and `metrics.csv`.

### 7.2 How to run

- **GUI:**  
  `python gui.py`

- **Experiments:**  
  `python -m analysis.run_experiments`

Use the generated figures and CSV data to populate the Results and Discussion section
with your own plots and numerical values.
