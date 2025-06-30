from Utils import Utils
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import general_utils

# Seeding for reproducibility
tf.config.experimental.enable_op_determinism()
SEED = 42
np.random.seed(SEED)          # NumPy operations
tf.random.set_seed(SEED)      # TensorFlow operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def run_model(L_val, EI_val, q_0, filename, nlayers, nnodes, epochs, capture, strategy=True):
    """
    Run the model with the given parameters.
    """
    utils = Utils(L_val, EI_val, q_0)
    W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, model, lambdas, losses = utils.train(nlayers=nlayers, nnodes=nnodes, epochs=epochs, capture=capture, resample=False, strategy=strategy) 
    general_utils.save_results(filename, W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, lambdas, losses)

# create an init function that is only run as run from this file
if __name__ == "__main__":
        
    nlayers = 2
    nnodes = 32
    epochs = 1001
    capture = 100
    
    # Parameter declaration
    L_val = 10
    EI_val = 20.83333
    q_0 = 0.015
    filename = "pinn_results"
    run_model(L_val, EI_val, q_0, filename, nlayers, nnodes, epochs, capture, strategy=False)

    print("Training of model completed.")