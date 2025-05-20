

# Physics-Informed Neural Networks (PINNs) for Bernoulli Beam Equation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A TensorFlow implementation of Physics-Informed Neural Networks (PINNs) for solving the Bernoulli beam bending equation with ReLoBRaLo adaptive loss balancing.

## Key Features

- **Physics-Informed Learning**: Solves EI·d⁴w/dx⁴ = 0 with boundary conditions
- **Adaptive Loss Weighting**: Implements ReLoBRaLo algorithm for robust training
- **Reproducible Research**: Full seed control for deterministic results
- **Visualization Tools**: Automatic plotting of solutions and training metrics

## Installation

```bash
git clone https://github.com/iagr98/PINN_IAM
```

**Requirements**:
- Python 3.8+
- TensorFlow 2.8+
- NumPy
- Matplotlib
- SymPy

## Usage

# PINNs for Beam Bending - Straight to the Point

This is a Physics-Informed Neural Network (PINN) that learns how beams bend under load. No fancy formatting, just what each file does:

## The Files Explained

### `Utils.py` - The Brains
This is where all the heavy lifting happens:
- Contains the actual neural network that learns the physics
- Handles the beam physics (that `EI·d⁴w/dx⁴ = 0` stuff)
- Manages the adaptive loss balancing (ReLoBRaLo magic)
- Does all the training logic
- Basically the "engine" of the whole project

Key things it does:
- Creates the neural network architecture
- Computes derivatives for the physics loss
- Implements boundary conditions (like fixed ends)
- Contains the training loop with plateau detection

### `general_utils.py` - The Organizer
This file takes care of:
- **Saving results** (numpy arrays of predictions, losses, etc.)
- **Plotting** (comparisons between predicted and exact solutions)
- **Loading data** (when you want to analyze results later)

It's useful because:
- Keeps all your results organized in folders
- Generates those nice loss curves and comparison plots
- Lets you replay results without retraining

### `main.py` - The Controller
This is where you actually run things:
- Sets up the beam problem (length, stiffness, load)
- Calls the trainer from `Utils.py`
- Decides where to save results
- Turns plotting on/off

Typical usage:
```python
# Set up a 10m beam with 333N load
run_model(F_val=333.33, L_val=10, EIz_val=111.11e6, 
          filename="my_beam_results",
          epochs=5000)
