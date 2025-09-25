

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
- Manages the adaptive loss balancing
- Does all the training logic
- Basically the "engine" of the whole project

Key things it does:
- Creates the neural network architecture
- Computes derivatives for the physics loss
- Implements boundary conditions (like fixed ends)
- Contains the training loop with plateau detection


### `main.py` - The Controller
This is where you actually run things:
- Sets up the beam problem (length, stiffness, load)
- Calls the trainer from `Utils.py`
- Decides where to save results
- Turns plotting on/off

Typical usage:

Type on the terminal

python main.py
