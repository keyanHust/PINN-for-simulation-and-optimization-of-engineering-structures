# Paper name: Numerical Simulation and Optimization of Cable-stayed Bridge Based on Physics-Informed Neural Networks (PINN)
Journal: Computers & Structures
Description: A PINN surrogate model for numerical simulation and optimization of cable-stayed bridges under static and dynamic loads.

## Overview

A two-step strategy is adopted:

1. **Surrogate Model Training** — A neural network takes structural design parameters (self-weight of girder and pylon, cable pre-tensioning forces) as input and outputs nodal displacements. Training is driven purely by the residuals of structural equilibrium equations, with no labeled data. A hybrid Adam + LBFGS optimizer is used.

2. **Structural Parameter Optimization** — The trained surrogate model is frozen, and the design parameters are optimized via gradient descent to achieve structural objectives such as minimum displacement or minimum bending energy.

## Key Features

- **Physics-driven loss**: The loss function consists solely of the residuals of the structural equilibrium equations. A dimensionless approach is employed by multiplying both sides by K⁻¹ to address magnitude disparities.
- **Hard boundary conditions**: Dirichlet BCs are enforced via output masking.
- **Constrained sampling**: A prior constraint ensures approximate balance between cable forces and gravity during training sampling.
- **Transferable**: The surrogate model functions as a "digital twin" capable of real-time prediction for any parameter combination within the design domain.

## File Structure

| File | Description |
|------|-------------|
| `CSB-Surrogate.py` | Main script: geometry, system assembly, network, training, and optimization |
| `requirements.txt` | Python dependencies |

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- SciPy

## Usage

```bash
pip install -r requirements.txt
python CSB-Surrogate.py
```

