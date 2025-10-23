from Utils import Utils


def run_model(epochs, scheme, conditioning, inverse, filename):
    utils = Utils(epochs=epochs, scheme=scheme, conditioning=conditioning, inverse=inverse)
    utils.train()
    utils.plot_save_results(path='Outputs/pinn_results_dynamic', filename=filename)

# create an init function that is only run as run from this file
if __name__ == "__main__":
    
    epochs = 31


    filename = 'dynamic_bernoulli.svg'
    run_model(epochs=epochs, scheme=None, conditioning=False, inverse=False, filename=filename)


    print("Training of model completed.")