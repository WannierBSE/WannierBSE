<div align="center">
  <img src="pics/logo_light.svg#gh-light-mode-only" alt="WannierBSE Logo" width="700" height="auto">
  <img src="pics/logo_dark.svg#gh-dark-mode-only" alt="WannierBSE Logo" width="700" height="auto">

# WannierBSE v1.1.1

<em>A Multiscale Exciton Simulation Package</em>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![MATLAB](https://img.shields.io/badge/MATLAB-Supported-blue.svg)](https://www.mathworks.com/products/MATlab.html)
[![Python](https://img.shields.io/badge/Python-Supported-blue.svg)](https://www.python.org/)
[![GitHub Stars](https://img.shields.io/github/stars/WannierBSE/WannierBSE?style=social)](https://github.com/WannierBSE/WannierBSE)

[**Overview**](#overview) | [**Key Features**](#key-features) | [**Available Implementations**](#available-implementations) | [**Getting Started & Examples**](#getting-started--examples) | [**Citation**](#citation) | [**Credits**](#credits) | [**Supported By**](#supported-by)
</div>

---

## Overview

WannierBSE is a high-performance computational software package for solving the Bethe–Salpeter equation (BSE) within the Wannier tight-binding (WTB) framework, specifically designed for two-dimensional (2D) materials. It is available in both **MATLAB and Python** for investigating excitonic and optical properties.

WannierBSE provides a multiscale framework that bridges microscopic electronic-structure information obtained from first-principles calculations with macroscopic dielectric screening models. This approach enables the simulation of excitonic properties in different dielectric environments, including free-standing, substrate-supported, and encapsulated 2D materials.

In addition to first-principles-based calculations, WannierBSE supports user-defined tight-binding electronic models for BSE calculations **when only the direct Coulomb interaction is considered**, allowing researchers to investigate excitonic properties without requiring Wannier functions generated from first-principles calculations.

Detailed instructions, input requirements, and computational examples are provided in the **User Guides and QuickStart Guides** for both implementations.

## Key Features

- **Multiscale Exciton Simulations:** Integrates first-principles electronic structures within the Wannier tight-binding framework with continuum dielectric screening models to investigate excitons in 2D materials.

- **Flexible Electronic-Structure Inputs:** Supports Wannier90-derived Hamiltonians and user-defined tight-binding models, with external electronic models available for direct-interaction-only calculations.

- **Nonlocal Dielectric Screening:** Incorporates momentum-dependent dielectric screening through a built-in Poisson solver for layered heterostructures or user-supplied dielectric functions.

- **Direct and Short-Range Exchange Interactions:** Constructs the BSE Hamiltonian including screened electron–hole attraction and optional short-range exchange interactions evaluated from real-space spinor Wannier functions.

- **Nonuniform Momentum-Space Sampling:** Generates momentum grids with refined sampling around the K and K′ valleys to improve resolution in exciton calculations.

- **Excitonic Eigenstates:** Computes exciton energies and momentum-space wavefunctions, providing numerical results for further excitonic and optical analysis.

## Available Implementations

| | MATLAB | Python |
|---|---|---|
| **Version** | v1.1.1 | v1.1.1 |
| **Main program** | `WBSE.m` | `WBSE.py` |
| **Software requirements** | MATLAB with Parallel Computing Toolbox | Python with required scientific libraries |
| **Source code** | [MATLAB Package](WannierBSE-MATLAB_1_1_1/) | [Python Package](WannierBSE-Python_1_1_1/) |
| **Documentation** | [User Guide](WannierBSE-MATLAB_1_1_1/WannierBSE_MATLAB_v1_1_1_User_Guide.pdf) | [User Guide](WannierBSE-Python_1_1_1/WannierBSE_Python_v1_1_1_User_Guide.pdf) |
| **QuickStart** | [MATLAB QuickStart](WannierBSE-MATLAB_1_1_1/WannierBSE-MATLAB_v1.1.1_QuickStart_Guide.pdf) | [Python QuickStart](WannierBSE-Python_1_1_1/WannierBSE-Python_v1.1.1_QuickStart_Guide.pdf) |

For implementation-specific instructions, see the [MATLAB README](WannierBSE-MATLAB_1_1_1/README.md) and [Python README](WannierBSE-Python_1_1_1/README.md).

## Getting Started & Examples

To begin, select your preferred implementation and follow its **QuickStart Guide** for installation, configuration, and execution instructions.

Both implementations include **seven example calculations**, covering different dielectric environments, user-defined tight-binding models, and direct Coulomb interactions with or without short-range exchange.

- [MATLAB Examples](WannierBSE-MATLAB_1_1_1/Examples/)
- [Python Examples](WannierBSE-Python_1_1_1/Examples/)

For detailed information on the theoretical framework, input-file formats, simulation parameters, and output data, refer to the corresponding **User Guide**.

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
  <img src="pics/NSTC-Light.webp#gh-light-mode-only" height="40" align="left">
  <img src="pics/NSTC-Dark.webp#gh-dark-mode-only" height="40" align="left">
  
  <img src="pics/TRIAD%20LIGHT%20INNOVATION-Light.webp#gh-light-mode-only" height="45" align="right">
  <img src="pics/TRIAD%20LIGHT%20INNOVATION-Dark.webp#gh-dark-mode-only" height="45" align="right">
  
  <img src="pics/NCHC-Light.webp#gh-light-mode-only" height="40">
  <img src="pics/NCHC-Dark.webp#gh-dark-mode-only" height="40">
</div>
<br clear="all">

---
<div align="center">
  Copyright © 2026 The WannierBSE Development Team. All rights reserved.
</div>
