from Utils import Utils


def run_model(epochs, scheme, conditioning, inverse, filename):
    utils = Utils(epochs=epochs, scheme=scheme, conditioning=conditioning, inverse=inverse)
    utils.train()
    utils.plot_save_results(path='Outputs/pinn_results30.09', filename=filename)

# create an init function that is only run as run from this file
if __name__ == "__main__":
    
    epochs = 300001

    filename = 'inverse_no_conditioning.svg'
    run_model(epochs=epochs, scheme='own', conditioning=False, inverse=True, filename=filename)

    filename = 'inverse_conditioning.svg'
    run_model(epochs=epochs, scheme='own', conditioning=True, inverse=True, filename=filename)

    filename = 'inverse_no_scheme.svg'
    run_model(epochs=epochs, scheme=None, conditioning=False, inverse=True, filename=filename)

    filename = 'inverse_relobralo.svg'
    run_model(epochs=epochs, scheme='relobralo', conditioning=False, inverse=True, filename=filename)


    print("Training of model completed.")