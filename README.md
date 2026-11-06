# 🚀 ASTRA: Aerospace System Transfer Reliability Analysis

**Cross-domain Degradation Transfer Learning Framework for Spacecraft Critical Components**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Electron](https://img.shields.io/badge/Electron-Desktop-47848F?logo=electron)](https://www.electronjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Project Overview

ASTRA is a rigorous cross-domain transfer learning framework designed for **Remaining Useful Life (RUL) prediction** of spacecraft critical components. Developed for the *"Aerospace System Transfer Reliability Analysis"* competition hosted by the **Innovation Academy for Microsatellites, Chinese Academy of Sciences (CAS)**, this project addresses the challenge of predicting RUL with extremely limited target-domain data.

The framework successfully transfers degradation knowledge from the source domain (nozzle ablation) to target domains (reaction wheels and batteries) by combining physics-informed gray-box models with unsupervised domain adaptation. It is accompanied by an **Electron desktop dashboard** that turns raw experiment outputs into an interactive, visualizable engineering workflow.

## 🌟 Key Features & Innovations

*   **Cross-Domain Transfer (PEUDA):** Utilizes dual time-frequency domain adaptation and contrastive learning to align distributions between distinct mechanical systems.
*   **Physics-Informed Gray-Box Model:** Embeds the Arrhenius degradation equation directly into the neural network architecture to ensure physical consistency.
*   **Self-Supervised Pretraining (SHRDL):** Employs Health State Decomposition Learning to extract robust features from unlabeled data, significantly improving few-shot transfer capability.
*   **Uncertainty Quantification (PCBNN):** Integrates a Physics-Constrained Bayesian Neural Network to provide reliable confidence intervals for predictions.
*   **Desktop Engineering Dashboard:** Electron + React application for interactive visualization of training curves, RUL trajectories, uncertainty bands, and ablation studies.

## 🏆 Benchmark Results (Selected)

*Protocol Note: All results below are evaluated under strict leakage-free protocols (e.g., participant/unit-disjoint splits, train-only normalization, no test-label early stopping).*

### 1. N-CMAPSS Dataset (Aero-engine RUL)
Strict protocol, 5-seed ensemble, `stride=1000`. Neural networks show significant superiority over classical baselines.

| Dataset | Selected Model | Neural RMSE | Ridge RMSE | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| DS01 | BiGRU (Full) | **7.208** | 13.068 | **+44.8%** |
| DS02 | BiGRU (Phys) | **6.495** | 11.269 | **+42.4%** |
| DS04 | BiGRU (Full) | **7.150** | 19.829 | **+63.9%** |
| DS08a | BiGRU (Phys) | **11.424** | 14.009 | **+18.5%** |
| **Macro** | **DS01-DS08** | **8.669** | 14.638 | **+40.8%** |

### 2. FEMTO → COMSOL Cross-Domain Transfer
Strict outer-LOO transfer from physical bearings to simulated reaction wheels.

| Transfer Variant | Transfer RMSE | Matched Scratch | Improvement |
| :--- | :--- | :--- | :--- |
| N=2 (Huber weight) | **4.974** | 5.366 | **+7.30%** |
| N=3 (fit-only inner-LOO)| **6.149** | 6.577 | **+6.51%** |
| Aggregated (N=1,2,3) | **6.233** | 6.664 | **+6.46%** |

### 3. Multi-condition Nozzle Ablation Benchmark
Simulated via a reduced-order ODE model calibrated by COMSOL (20s trajectory error <0.5%).

| Evaluation Tier | Selected Neural RMSE | Best Classical | Notes |
| :--- | :--- | :--- | :--- |
| **Estimated** (depth+rate)| **2.01 s** | RF: 0.40 s | Extremely low error under strict leakage check |
| **Physics-Obs** (proxy)| **37.90 s** | Huber: 27.29 s | Proxies introduce noise; classical models are more robust |
| **Observable** (T/Q/P)| **38.15 s** | Ridge: 28.99 s | Baseline benchmark |
## 🏗️ System Architecture

```text
ASTRA/
├── src/                      # Core Python package
│   ├── models/               # Core Models (SHRDL, PEUDA, PCBNN, TCN, BiGRU)
│   ├── data/                 # Data loaders (C-MAPSS, NASA, COMSOL, FEMTO)
│   ├── baselines/            # Classical ML (Ridge, Huber, GBDT, RF)
│   ├── evaluate/             # Strict evaluation metrics & protocols
│   ├── uncertainty/          # Bayesian uncertainty estimation
├── scripts/                  # End-to-end pipelines
│   ├── exp_ncmapss_strict.py         # N-CMAPSS rigorous benchmark
│   ├── exp_femto_to_comsol_v5.py     # Cross-domain transfer v5
│   ├── run_nozzle_sota.py            # Multi-trajectory nozzle benchmark
│   ├── exp_main_suite.py             # Full competition suite
├── frontend/                 # 🖥️ Electron + React desktop dashboard
│   ├── src/main/             #   Electron main process
│   ├── src/preload/          #   Preload bridge
│   ├── src/renderer/         #   React + Tailwind UI
│   ├── src/shared/           #   Shared types
├── simulation/               # 🧪 COMSOL physical modeling docs (added back)
│   ├── nozzle_ablation_20s/  #   ReactingFlowSST + ALE moving mesh throat ablation
│   ├── geo_battery_pinn_extreme/  # PINN for GEO satellite battery degradation
│   ├── reaction_wheel_em_parker/  # Parker EM model for reaction wheel
├── docker/                   # One-click reproduction environment
├── exp_*.py                  # Individual experiment entry points
├── tests/                    # Unit tests for data leakage and stability
├── docs/                     # Supporting documentation & audits
├── ASTRA_*.docx              # Judge-facing technical reports
```

## 🔧 Quick Start & Reproduction

### 1. Environment Setup

```bash
conda create -n astra python=3.10 -y
conda activate astra
pip install -r requirements.txt
```

### 2. One-Click Docker Reproduction

We provide a fully reproducible Docker environment for the core FEMTO → COMSOL transfer experiment.

```powershell
# Windows
.\scripts\reproduce_femto_ims_to_comsol.ps1

# Linux/macOS
./scripts/reproduce_femto_ims_to_comsol.sh
```

*The script automatically validates data SHA-256 hashes before execution to prevent tampering.*

### 3. Run Key Benchmarks

```bash
# N-CMAPSS Benchmark
python scripts/exp_ncmapss_strict.py --data-dir path/to/data --stride 1000 --epochs 200

# Multi-trajectory Nozzle SOTA
python scripts/run_nozzle_sota_validation.py \
    --data data/processed/nozzle_mt_200/nozzle_sim_200traj.csv \
    --window-size 20 --epochs 160 --device cuda:0
```

### 4. Launch the Desktop Dashboard

```bash
cd frontend
npm install
npm run dev         # Run both Vite dev server and Electron main process
# or
npm run build       # Package into a production executable (dist-electron/)
```

The dashboard connects to the Python backend via an IPC bridge and provides interactive visualizations of:
- Training curves & ablation trajectories
- RUL prediction trajectories with uncertainty bands
- Transfer-learning heatmaps (FEMTO → COMSOL)
- One-click benchmark runner for all competition experiments

## 🔬 Engineering Safeguards

This project strictly adheres to high-standard engineering and scientific research protocols:

*   **Data Leakage Prevention:** All experiments use unit-disjoint or participant-disjoint splits. Test sets are evaluated exactly once.
*   **Numerical Stability:** `clamp(min=1e-6)` for pooling, grouped weight decay, and Z-score/inverse evaluation.
*   **Shortcut Prevention:** Interviewer prompt filters (for NLP tasks), minimum audio filters, and strictly no test-label early stopping.

## 📝 Acknowledgements

*   **Competition Host:** Innovation Academy for Microsatellites, Chinese Academy of Sciences.
*   **Datasets:** NASA C-MAPSS, N-CMAPSS, FEMTO/IMS, RAVDESS.
*   **Frameworks:** PyTorch, Scikit-Learn, SciPy, Electron, React.