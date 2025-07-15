from Utils import Utils
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import general_utils

# Seeding for reproducibility
SEED = 42
tf.random.set_seed(SEED)      # TensorFlow operations


def run_model(L_val, EI_val, q_0, filename, nlayers, nnodes, epochs, capture, strategy=True):
    """
    Run the model with the given parameters.
    """
    utils = Utils(L_val, EI_val, q_0)
    W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, model, lambdas, losses = utils.train(nlayers=nlayers, nnodes=nnodes, epochs=epochs, capture=capture, resample=False, strategy=strategy) 
    general_utils.save_results(filename, W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, lambdas, losses)

# create an init function that is only run as run from this file
if __name__ == "__main__":
        
    nlayers = 4
    nnodes = 16
    epochs = 10001
    capture = 1000
    
    # Parameter declaration
    L_val = 10
    EI_val = 20.83333
    q_0 = 0.015
    filename = "pinn_results_seeded"
    run_model(L_val, EI_val, q_0, filename, nlayers, nnodes, epochs, capture, strategy=False)

    print("Training of model completed.")