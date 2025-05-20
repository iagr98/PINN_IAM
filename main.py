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


def run_model(F_val, L_val, EIz_val, filename, epochs, capture, strategy=True):
    """
    Run the model with the given parameters.
    """
    utils = Utils(F_val, L_val, EIz_val)
    W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, model, lambdas, losses = utils.train(epochs=epochs, capture=capture, resample=False, strategy=strategy) 
    general_utils.save_results(filename, W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, lambdas, losses)

# create an init function that is only run as run from this file
if __name__ == "__main__":
    
    # Parameter declaration
    epochs = 10001
    capture = 100
    F_val=333.333333333
    L_val =10
    EIz_val = 111.111e6
    
    # Train the model 1 without strategy
    filename = "pinn_results_no_strategy"
    run_model(F_val, L_val, EIz_val, filename, epochs, capture, strategy=False)

    # Train the model 2 with strategy
    filename = "pinn_results_strategy"
    run_model(F_val, L_val, EIz_val, filename, epochs=epochs, capture=capture, strategy=True)

    print("Training of both models completed.")
        