# PINN-Based Cosmic Structure Formation

Physics-Informed Neural Networks for Modeling Cosmic Structure Formation via the Vlasov–Poisson System


## Overview

This project investigates the use of Physics-Informed Neural Networks (PINNs) to model cosmic structure formation by solving the Vlasov–Poisson equations, which govern the evolution of collisionless self-gravitating matter such as dark matter.

Instead of relying on traditional N-body simulations, which are computationally expensive and scale poorly with resolution and dimensionality, this project explores whether neural networks constrained by physical laws can serve as accurate and efficient continuous surrogate solvers for cosmological partial differential equations.

This work lies at the intersection of:

- Scientific Machine Learning  
- Computational Cosmology  
- Physics-Informed Deep Learning  
- PDE-Constrained Optimization  

## Objectives

- Implement a Physics-Informed Neural Network for the Vlasov–Poisson system  
- Learn the phase-space distribution function of collisionless matter  
- Enforce physical laws directly through PDE residual minimization  
- Study the feasibility of PINNs as surrogate solvers for cosmic structure formation  
- Build a research-grade, extensible baseline for further work  


## Scientific Background

### Cosmic Structure Formation

Large-scale structures such as galaxies, filaments, and voids arise due to gravitational instability in an initially nearly homogeneous matter distribution.  
For collisionless matter (e.g., dark matter), this evolution is accurately described by the Vlasov–Poisson equations.


### Vlasov–Poisson System

The system consists of:

#### Vlasov Equation

\[
\frac{\partial f}{\partial t}
+ v \cdot \nabla_x f
- \nabla_x \phi \cdot \nabla_v f
= 0
\]

#### Poisson Equation

\[
\nabla^2 \phi = 4\pi G \rho
\]

with matter density defined as:

\[
\rho(x, t) = \int f(x, v, t)\, dv
\]

---

## Why Physics-Informed Neural Networks?

Traditional solvers:
- Require large particle counts or fine grids  
- Are expensive for long time evolution  
- Do not generalize across initial conditions  

Physics-Informed Neural Networks:
- Represent the solution as a neural network  
- Enforce PDEs via automatic differentiation  
- Require no labeled simulation data  
- Provide continuous solutions in space and time  

This makes them attractive as surrogate solvers and analysis tools for complex physical systems.

---

## Methodology

### Neural Network Representation

A neural network \( f_\theta \) approximates the phase-space distribution function.

Input:
- Spatial coordinates \( x \)  
- Velocity / momentum \( v \)  
- Time \( t \)  

Output:
- Distribution function \( f(x, v, t) \)


### Physics-Informed Loss Function

The total loss is a weighted sum of:

- Vlasov equation residual  
- Poisson equation residual  
- Initial condition loss  
- Boundary condition loss (if applicable)

\[
\mathcal{L} =
\lambda_1 \mathcal{L}_{\text{Vlasov}} +
\lambda_2 \mathcal{L}_{\text{Poisson}} +
\lambda_3 \mathcal{L}_{\text{IC}} +
\lambda_4 \mathcal{L}_{\text{BC}}
\]

All derivatives are computed using automatic differentiation.

---

## Repository Structure

```text
PINN-Cosmic-Structure-Formation/
│
├── pinn/
│   ├── model.py          # PINN architecture
│   ├── loss.py           # PDE residual definitions
│   └── trainer.py        # Training loop
│
├── simulations/
│   ├── initial_conditions.py
│   ├── domain.py
│   └── run_training.py
│
├── results/
│   ├── density_fields/
│   ├── loss_curves/
│   └── visualizations/
│
├── requirements.txt
└── README.md
