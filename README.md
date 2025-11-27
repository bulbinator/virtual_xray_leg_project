# Virtual X-Rays of a Simplified Leg Phantom

This project implements a simple computational simulator of a conventional X-ray system
imaging a simplified **leg phantom** made of two concentric cylinders (soft tissue and bone).

Main components:

- 3D cylindrical leg phantom generator (`phantom.py`)
- Parallel-beam X-ray forward model with kVp, distance, noise and blur (`xray_simulator.py`)
- Image quality metrics: MSE, SSIM, Gradient Error, Gradient Entropy (`metrics/metrics.py`)
- Interactive GUI to generate phantoms, run simulations and view metrics (`gui.py`)
- Scripted experiments to sweep parameters and save results (`analysis/run_experiments.py`)
- Example report template (`report/Final_Report_Virtual_XRays_LegPhantom.md`)

---

## 1. Installation

### 1.1. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
