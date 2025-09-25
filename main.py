from Utils import Utils


def run_model(epochs, scheme, conditioning, filename):
    utils = Utils(epochs=epochs, scheme=scheme, conditioning=conditioning)
    utils.train()
    utils.plot_save_results(path='Outputs/pinn_results26.09', filename=filename)

# create an init function that is only run as run from this file
if __name__ == "__main__":
    
    epochs = 300001

    filename = 'no_conditioning.svg'
    run_model(epochs=epochs, scheme='own', conditioning=False, filename=filename)

    filename = 'conditioning.svg'
    run_model(epochs=epochs, scheme='own', conditioning=True, filename=filename)

    filename = 'no_scheme.svg'
    run_model(epochs=epochs, scheme=None, conditioning=False, filename=filename)


    print("Training of model completed.")