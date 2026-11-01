<div align="center">
  <img src="../pics/logo_light.svg#gh-light-mode-only" alt="WannierBSE Logo" width="700" height="auto">
  <img src="../pics/logo_dark.svg#gh-dark-mode-only" alt="WannierBSE Logo" width="700" height="auto">

# WannierBSE-MATLAB v1.1.1

[Home](../README.md)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![MATLAB](https://img.shields.io/badge/MATLAB-Supported-blue.svg)](https://www.mathworks.com/products/MATlab.html)
[![GitHub Stars](https://img.shields.io/github/stars/WannierBSE/WannierBSE?style=social)](https://github.com/WannierBSE/WannierBSE)

[**Getting Started**](#getting-started) | [**Technical Workflow**](#technical-workflow) | [**Examples**](#examples) | [**Citation**](#citation) | [**Credits**](#credits) | [**Supported By**](#supported-by)
</div>

---

**WannierBSE** is a high-performance computational suite for solving the **Bethe–Salpeter Equation (BSE)** within a Wannier tight-binding framework. Specifically optimized  for two-dimensional materials, it combines first-principles electronic-structure accuracy with the efficiency of Wannier-based interpolation.

The general workflow begins with a density functional theory (DFT) calculation performed using Quantum ESPRESSO, VASP, or another first-principles code compatible with Wannier90. The DFT results are then processed with Wannier90 to construct a localized Wannier representation of the electronic structure. **WannierBSE** uses the generated Wannier Hamiltonian and related matrix elements to construct and solve the BSE.

By leveraging MATLAB’s numerical capabilities, WannierBSE enables researchers to calculate exciton energies, excitonic wavefunctions, optical spectra, transition dipoles, and light–matter interactions in complex dielectric environments.

## Getting Started

To get started with WannierBSE, please refer to our comprehensive documentation:

*   [**User Guide (PDF)**](WannierBSE_MATLAB_v1_1_1_User_Guide.pdf) - Detailed technical manual and theory.
*   [**QuickStart Guide (PDF)**](WannierBSE-MATLAB_v1.1.1_QuickStart_Guide.pdf) - Brief setup and first run instructions.
*   [**Official Website**](https://quantum.web.nycu.edu.tw/wannierbse) - News and updates.

### Prerequisites
*   **MATLAB** (Recommended: R2020b or later)
*   **Parallel Computing Toolbox** (Highly recommended for large-scale calculations)

## Technical Workflow

WannierBSE features a modular data pipeline designed to handle both raw generation and pre-computed data ingestion.

### 1. Data Preparation
WannierBSE supports direct-interaction calculations and, in v1.1.1, calculations including short-range electron-hole exchange.

> 🔄 **Direct-Interaction Route:** Users may either place `wannier90_hr.dat` in `/User_input/` and let WannierBSE generate the k-mesh and tight-binding bands internally, or provide a matched external dataset using `kmesh.txt` and `TB_data/` files such as `v*_TB.txt` and `c*_TB.txt`. Structural information is read from `structure.txt` or `wannier90.win` in `/User_input/`.
> 
> 📥 **Direct + Short-Range Exchange Route:** Exchange-enabled calculations require the internal Wannier90-based workflow. In addition to `wannier90_hr.dat`, users must provide real-space spinor Wannier functions in `/User_input/Wannier_functions_xsf/`, unless the required processed Wannier-function and exchange-interaction caches already exist in `/Precomputed_data/`.
>
> Examples 05–07 include short-range exchange. Due to GitHub file-size limitations, the real-space Wannier-function datasets and their processed caches are not included in the source distribution.
>
> **Dielectric Input:** For both routes, users may provide `epsilon.txt` in `/User_input/`; otherwise WannierBSE generates or loads dielectric data from `/Precomputed_data/`.

### 2. Configuration & Control
The simulation environment is governed by dedicated control files located in the `/Parameters/` directory:
*   `control.txt`, `WTB_control.txt`, `WF_centers.txt`
*   `kmesh_control.txt`, `dielectric_control.txt`
*   `exchange_control.txt`

### 3. Execution & Visualization
1.  **Core Solver:** Execute `WBSE.m` in MATLAB. The script utilizes an optimized, symmetry-aware Hamiltonian constructor and a high-performance parallelized solver.
2.  **Post-Processing:** Use the auxiliary script `Ex_plot.m` to visualize the exciton energy spectrum.
3.  **Data Management:** Main outputs, including energy spectra (`Ex.mat`) and wavefunctions (`A.mat`), are saved in the `/Exciton_data/` folder together with run-related output files.

## Examples

To help you get started quickly, we provide several pre-configured examples in the repository. These include sample input files and parameters for direct-interaction and direct-plus-short-range-exchange workflows in typical 2D materials.

📂 [**Browse Examples in the Repository**](Examples/)

Each example includes:
*   User input and precomputed data files
*   Ready-to-use control parameter files
*   Expected results.

## Citation

To support the continued development of WannierBSE, please cite the following references:

> **Primary Reference:**
> Peng, G.-H. et al. "Distinctive Signatures of the Spin- and Momentum-Forbidden Dark Exciton States in the Photoluminescence of Strained WSe2 Monolayers under Thermalization", 
> *Nano Lett.* 19, 4, 2299–2312 (2019). 
> [https://doi.org/10.1021/acs.nanolett.8b04786](https://doi.org/10.1021/acs.nanolett.8b04786)

> **Non-local Dielectric Function:**
> Li, W.-H. et al. "The Key Role of Non-Local Screening in the Environment-Insensitive Exciton Fine Structures of Transition-Metal Dichalcogenide Monolayers", 
> *Nanomaterials* 13, 11, 1739 (2023). 
> [https://doi.org/10.3390/nano13111739](https://doi.org/10.3390/nano13111739)

## Credits

Developed by the research group of **Prof. Shun-Jen Cheng** at National Yang Ming Chiao Tung University (NYCU). [**Official Website**](https://quantum.web.nycu.edu.tw/wannierbse)

**Core Developers:**
*   Dr. Ping-Yuan Lo, Dr. Wei-Hua Li, Dr. Guan-Hao Peng, Dr. Jhen-Dong Lin, Dr. Vo Khuong Dien, Dr. Oscar Javier Gomez Sanchez, Mr. Ching-Hung Shih, Mr. Kun-Yi Lin, and Prof. Shun-Jen Cheng.

## Supported By

We gratefully acknowledge the support from the following organizations:

<div align="center">
  <img src="../pics/NSTC-Light.webp#gh-light-mode-only" height="40" align="left">
  <img src="../pics/NSTC-Dark.webp#gh-dark-mode-only" height="40" align="left">
  
  <img src="../pics/TRIAD%20LIGHT%20INNOVATION-Light.webp#gh-light-mode-only" height="45" align="right">
  <img src="../pics/TRIAD%20LIGHT%20INNOVATION-Dark.webp#gh-dark-mode-only" height="45" align="right">
  
  <img src="../pics/NCHC-Light.webp#gh-light-mode-only" height="40">
  <img src="../pics/NCHC-Dark.webp#gh-dark-mode-only" height="40">
</div>
<br clear="all">

---
<div align="center">
  Copyright © 2026 The WannierBSE Development Team. All rights reserved.
</div>
