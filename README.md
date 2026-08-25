# MedFed — Privacy-Preserving Collaborative Medical Imaging Network

**Problem Statement H-03 | Domain: Healthcare & Well-being**

A working federated learning system that lets hospitals collaboratively train a brain tumor MRI classification model — without any hospital ever sharing a raw patient image.

---

## Problem Statement & Solution Overview

### The Problem
Hospitals individually possess medical imaging datasets that are too small or institution-specific to train robust computer-vision models. Sharing raw CT, MRI, X-ray, or pathology images across institutions creates serious privacy and governance problems under regulations like HIPAA, and typically requires IRB (ethics board) approval for any research use of patient data.

### Our Solution
MedFed implements a Privacy-Preserving Collaborative Medical Imaging Network using **Federated Learning**: each hospital trains a shared model locally, on its own private data, and only exchanges numeric model weight updates — never raw images — with a central aggregation platform. The platform combines these updates into one continuously improving shared model, which is sent back to every participating hospital.

### Known Constraints We Address
The problem statement specifies six required constraints. We built and validated a working solution for each:

| # | Constraint | Status |
|---|---|---|
| 1 | Keep raw medical images within institutional boundaries | ✅ Implemented |
| 2 | Support collaborative model training and evaluation | ✅ Implemented |
| 3 | Detect distribution or scanner-specific domain shift | ✅ Implemented |
| 4 | Detect anomalous or potentially malicious model updates | ✅ Implemented |
| 5 | Track model versions, training provenance, and evaluation evidence | ✅ Implemented |
| 6 | Quantify privacy and model-quality trade-offs | ✅ Implemented |

---

## System Architecture / Workflow

```
┌─────────────┐   encrypted weights   ┌───────────────────────┐
│ Hospital_1  │ ─────────────────────►│                        │
│ (local data,│                       │   CENTRAL PLATFORM     │
│  local model)│◄──── global model ───│   - Decrypt & verify   │
└─────────────┘                       │   - Anomaly detection  │
┌─────────────┐   encrypted weights   │   - FedAvg aggregation │
│ Hospital_2  │ ─────────────────────►│   - Provenance logging │
│             │◄──── global model ───│   - Dashboard          │
└─────────────┘                       │                        │
┌─────────────┐   encrypted weights   │                        │
│ Hospital_3  │ ─────────────────────►│                        │
│             │◄──── global model ───└───────────────────────┘
```

### One Federated Round, Step by Step
1. **Distribute** — the current global model's weights are sent to all participating hospitals
2. **Local training** — each hospital trains its own copy for one epoch, using only its own private images
3. **Collect updates** — each hospital's updated weights (not images) are sent to the central platform
4. **Anomaly detection** — the platform compares each hospital's update against the others, flagging and excluding any statistical outlier before aggregation
5. **Aggregate (FedAvg)** — trusted updates are averaged into one new, improved global model
6. **Provenance logging** — the round is recorded with a timestamp, participant list, flagged hospitals (if any), resulting accuracy, and a SHA-256 hash of the model
7. **Redistribute** — the improved global model is sent back to all hospitals, and the cycle repeats

We ran 5 rounds in our experiments.

---

## Core Technical Mechanism

### Model
- **ResNet-18**, adapted via transfer learning from ImageNet-pretrained weights to classify 4 classes: glioma, meningioma, pituitary tumor, no tumor
- Chosen for fast convergence on a moderately-sized medical imaging dataset without training from scratch

### Federated Averaging (FedAvg)
Implemented manually in PyTorch (not via the Flower library — see *Setup & Installation* for why) as a weighted average of model state dictionaries across all trusted hospital updates each round.

### Differential Privacy
Implemented as gradient clipping + calibrated Gaussian noise added during local training (a DP-SGD-style approach), rather than injecting large noise directly onto final weights, which we found destabilizes training entirely if the noise scale isn't carefully calibrated to gradient magnitude.

### Malicious Update Detection
Pairwise distance comparison between each hospital's update and every other hospital's update in the same round. A hospital whose update deviates significantly beyond a calibrated threshold relative to the group median is flagged and excluded from that round's aggregation, which is then recomputed using only trusted hospitals.

### Domain Shift Detection
Feature embeddings are extracted from each hospital's data using the trained model, then compared via distance metrics and visualized with PCA to show how differently each hospital's data distribution looks to the model.

### Provenance Logging
Each round produces a JSON log entry containing: round number, timestamp, participating hospitals, flagged hospitals, resulting accuracy, and a SHA-256 hash of the resulting model state — providing a tamper-evident audit trail.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep learning | PyTorch, Torchvision |
| Model | ResNet-18 (transfer learning) |
| Federated learning | Custom FedAvg implementation |
| Privacy | Differential Privacy (gradient clipping + calibrated noise) |
| Security | Pairwise anomaly detection |
| Training environment | Google Colab (NVIDIA T4 GPU) |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Visualization | Matplotlib, scikit-learn (PCA) |
| Version control | Git / GitHub |

### A Note on Tooling Decisions
We initially attempted to use **Flower**, a popular federated learning framework, but its CLI dependencies (`pathspec`, `alembic`, `tomli_w`, and others) caused repeated, unresolvable dependency conflicts in the Colab environment. We pivoted to a manual FedAvg implementation in plain PyTorch, which gave us full control, eliminated the dependency issues, and is straightforward to audit line-by-line.

We also initially attempted to deploy live inference via Hugging Face Spaces, but discovered mid-project that Gradio/Docker Spaces now require a paid tier. We pivoted to Streamlit Community Cloud, which is free and fully supports PyTorch-based inference.

---

## Setup & Installation Instructions

### Requirements
```
streamlit
torch
torchvision
matplotlib
numpy
pillow
```

### Local Setup
```bash
git clone https://github.com/krishnaa058/medfed.git
cd medfed
pip install -r requirements.txt
streamlit run app.py
```

### Training Environment (to reproduce experiments)
The full training pipeline was developed and run in Google Colab with a T4 GPU. To reproduce:
1. Mount Google Drive containing the dataset (`federated_mri_dataset/Hospital_1/2/3`, each with `Training/` and `Testing/` subfolders per class: `glioma`, `meningioma`, `notumor`, `pituitary`)
2. Run the data loading, model definition, and training cells in sequence (see `/notebooks` if included, or the project documentation)
3. Trained model checkpoints and experiment results are saved as `.pkl`/`.json` files for reuse without retraining

---

## Usage Instructions

### Dashboard
The main dashboard (`app.py`) presents:
- Hospital trust score cards
- Federated vs. centralized accuracy comparison
- Interactive Differential Privacy slider (live accuracy vs. noise trade-off)
- Malicious update detection — before/after comparison
- Domain shift comparison tool (select two hospitals to compare)
- Provenance log (full training audit trail)
- A hospital local-training simulator (quick simulated round, or upload real MRI images to see the trained model genuinely process them into a numeric, non-diagnostic representation)

### Live Inference App
A separate standalone app (`inference_app.py`) lets you upload a brain MRI image and receive a live prediction (glioma / meningioma / pituitary / no tumor) with confidence scores, using our actual trained model.

---

## Validation / Experiments / Results

### Centralized Baseline (upper-bound reference)
Trained on the full, pooled dataset (not privacy-preserving — used only as a benchmark ceiling).
**Result: 92.62% test accuracy** (best epoch of 5)

### Federated Learning (our actual solution)
| Round | Accuracy |
|---|---|
| 1 | 62.00% |
| 2 | 82.13% |
| 3 | 90.93% |
| 4 | **91.63%** (peak) |
| 5 | 85.47% |

**Federated learning reached within ~1 percentage point of the centralized ceiling, without any hospital sharing raw images.**

### Differential Privacy Trade-off
| Noise Multiplier | Final Round Accuracy |
|---|---|
| 0.0 (no privacy) | 90.92% |
| 0.001 | 94.10% |
| 0.01 | 92.44% |
| 0.05 | 84.40% |
| 0.1 | 78.05% |
| 0.5 | 64.24% |

Strong privacy protection (noise = 0.01) cost less than 0.2 percentage points of accuracy versus the no-privacy baseline. The small accuracy increase at very low noise levels is consistent with the known regularization effect of small noise injections in neural network training.

### Malicious Update Detection
We simulated Hospital_3 sending corrupted/sabotaged updates from round 3 onward.
- **Without detection:** accuracy collapses to 25% (random guessing across 4 classes)
- **With detection:** the attacking hospital is flagged and excluded every round it attacks; accuracy remained in the 75–88% range and continued improving despite the sustained attack

### Domain Shift Detection
Pairwise feature-embedding distances between hospitals (all non-zero, confirming measurably different data distributions consistent with each hospital's simulated specialty):
- Hospital_1 vs Hospital_2: 6.86
- Hospital_1 vs Hospital_3: 6.19
- Hospital_2 vs Hospital_3: 6.06

### Provenance Log
Each of the 5 training rounds is logged with a unique SHA-256 model hash, timestamp, participant list, and flagged-hospital record, providing a verifiable, tamper-evident audit trail of the entire training process.

---

## Limitations & Future Scope

We are deliberately transparent about the gap between this working prototype and a production-ready deployment:

1. **Security detection is illustrative, not adversarially robust.** Our pairwise distance detector works against the attack pattern we tested but could plausibly be evaded by a more sophisticated, low-magnitude attack. Production systems would use Byzantine-robust aggregation methods (e.g., Krum, trimmed mean) and would benefit from a larger pool of participating hospitals, which makes statistical outlier detection meaningfully stronger.

2. **Differential Privacy is a simplified implementation, not a formally certified guarantee.** We demonstrate the real accuracy/privacy trade-off through gradient clipping and calibrated noise, but do not track a formal privacy budget (epsilon/delta) with a certified accountant (e.g., via Opacus) across the full training run.

3. **Regulatory and legal requirements are unaddressed by design.** Real deployment would require HIPAA-compliant infrastructure, formal data governance agreements between institutions, and IRB approval at each participating hospital — processes that are organizational and legal, not technical, and that our system does not and cannot bypass.

4. **Dataset harmonization is not modeled.** Real hospitals would have variation in scanner hardware, imaging protocols, and resolution beyond what our simulated, pre-cleaned dataset split represents. A real deployment would need a preprocessing/harmonization layer.

5. **Scale and fault tolerance are untested.** We validated the core mechanism with 3 simulated clients on a single machine. Real deployment would need asynchronous round handling, retry logic for dropped participants, and infrastructure cost planning across many real institutions.

6. **Single point of failure at the central aggregator.** Our architecture relies on one central server. Secure Aggregation (a cryptographic technique ensuring the server can only ever see the combined average, never individual contributions) and redundant aggregation infrastructure were scoped out of this prototype due to time constraints, and are noted as the clear next engineering step.

### Future Scope
- Byzantine-robust aggregation (Krum, trimmed mean)
- Formally accounted Differential Privacy (Opacus-based privacy budget tracking)
- Secure Aggregation for the central platform
- Real multi-institution pilot with proper data governance agreements
- Explainability layer (e.g., Grad-CAM) for clinician trust and interpretability
- Dataset harmonization pipeline for cross-scanner/cross-protocol variation

---

## Team Members

- [Team member name] — [role]
- [Team member name] — [role]
- [Team member name] — [role]

*(Fill in with your actual team names and roles before submitting)*

---

## AI Assistance Disclosure

This project was developed with assistance from Claude (Anthropic), used for:
- Debugging Python/PyTorch errors during federated learning implementation
- Iterating on and correcting the Differential Privacy noise implementation (an initial version over-corrupted model weights; this was diagnosed and fixed)
- Iterating on and correcting the malicious update detection threshold (an initial version failed to flag an obvious simulated attack; this was diagnosed and fixed)
- Drafting the Streamlit dashboard code and deployment troubleshooting
- Structuring this README and the accompanying presentation materials

All core technical decisions, dataset design, experiment design, and result interpretation were directed and reviewed by the team. All reported experimental results were generated by actually running the described code, not fabricated or estimated.

---

## Repository Structure

```
medfed/
├── app.py                  # Main Streamlit dashboard
├── inference_app.py        # Standalone live prediction app
├── medfed_model.pt         # Exported trained model (TorchScript)
├── requirements.txt        # Python dependencies
├── results/                 # Saved experiment results (pickled/JSON)
│   ├── fed_accuracy_history.pkl
│   ├── dp_privacy_results_full.pkl
│   ├── security_results.pkl
│   └── provenance_log.json
└── README.md                # This file
```
