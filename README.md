# ⚛️ Quantum Monte Carlo & Neural States Lab

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-NQS-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebooks-Research-F37626?logo=jupyter&logoColor=white)
![Physics](https://img.shields.io/badge/Domain-Quantum_Physics-purple)
![License](https://img.shields.io/badge/License-MIT-green)

> **"Solving the Schrödinger Equation... first with dice, then with brains."**

This repository documents a comprehensive journey through computational quantum physics. It progresses from traditional **Variational Monte Carlo (VMC)** methods to state-of-the-art **Neural Quantum States (NQS)** using Deep Learning, solving problems ranging from atomic structures to quantum phase transitions.

---

## 📖 Project Overview

The goal of this project is to find the **Ground State Energy** of various quantum systems by minimizing the expectation value of the Hamiltonian:

$$E_0 \le E(\theta) = \frac{\langle \Psi_\theta | \hat{H} | \Psi_\theta \rangle}{\langle \Psi_\theta | \Psi_\theta \rangle}$$

The project is divided into **6 Phases**, bridging the gap between classical computational physics and modern AI-driven research.

### 🏆 Key Achievements
* **Beating Hartree-Fock:** Achieved lower energy for the Helium atom by including electron correlation via Jastrow factors.
* **Computational Chemistry:** Simulated the H-H chemical bond and calculated the equilibrium bond length.
* **AI for Physics:** Implemented **Neural Network Quantum States (NQS)** using PyTorch to solve many-body problems without pre-defined physical ansatzes.
* **Quantum Phase Transition:** Simulated the critical point of the Transverse-Field Ising Model.

---

## 📂 Repository Structure

```text
vmc-quantum-lab/
├── 1_harmonic_oscillator/   # Phase 1: Basic VMC & Metropolis Algorithm
├── 2_helium_atom/           # Phase 2: Many-body Physics & Correlation
├── 3_hydrogen_molecule/     # Phase 3: Computational Chemistry (Bonding)
├── 4_nqs_harmonic/          # Phase 4: Intro to Neural Quantum States
├── 5_heisenberg_spin/       # Phase 5: Quantum Magnetism (Spin Chains)
├── 6_tfim_phase_transition/ # Phase 6: Quantum Phase Transitions (TFIM)
├── assets/                  # Images and plots
├── utils/                   # Shared utility functions
└── README.md
