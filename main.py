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


def run_model(L_val, EI_val, q_0, filename, epochs, capture, strategy=True):
    """
    Run the model with the given parameters.
    """
    utils = Utils(L_val, EI_val, q_0)
    W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, model, lambdas, losses = utils.train(epochs=epochs, capture=capture, resample=False, strategy=strategy) 
    general_utils.save_results(filename, W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, lambdas, losses)

# create an init function that is only run as run from this file
if __name__ == "__main__":
        
    epochs = 50001
    capture = 1000

    # Parameter declaration case 1
    L_val = 1
    EI_val = 1
    q_0 = 97.409
    filename = "pinn_results_case_1"
    run_model(L_val, EI_val, q_0, filename, epochs, capture, strategy=False)

    # Parameter declaration case 2
    L_val = 10
    EI_val = 1e7
    q_0 = 97.409
    filename = "pinn_results_case_2"
    run_model(L_val, EI_val, q_0, filename, epochs, capture, strategy=False)

    # Parameter declaration case 3
    L_val = 10
    EI_val = 1e7
    q_0 = 974.09
    filename = "pinn_results_case_3"
    run_model(L_val, EI_val, q_0, filename, epochs, capture, strategy=False)

    print("Training of model completed.")