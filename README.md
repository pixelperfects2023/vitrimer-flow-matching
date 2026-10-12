# Physics-Informed Flow Matching for Inverse Vitrimer Design

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pixelperfects2023/vitrimer-flow-matching/blob/main/latentflow.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Official repository for **Physics-Informed Latent Flow Matching for Inverse Vitrimer Design**. The model generates dynamic covalent linkers targeting specific Frontier Molecular Orbital (FMO) HOMO-LUMO energy gaps ($\Delta E_{\text{gap}} \approx 0.2037 \text{ Ha}$) using an ODE-based Continuous Normalizing Flow (CNF) along with nearest-neighbor inverse retrieval on QM9 dataset.

## Key Benchmark Results
- **Synthetic Validity:** 100% (Zero chemical hallucinations via manifold projection)
- **Top Physical Targeting Error:** $0.0104 \text{ Hartrees}$ ($\sim 0.283 \text{ eV}$) for candidate `c1cc(cnc1)C(=O)N` (Pyridine-carboxamide)
- **Physics Loss Reduction:** 99.18% decrease over 250 training epochs (from $1.6150$ down to $0.0133$)

## Repository Structure
