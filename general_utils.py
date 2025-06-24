import numpy as np
import matplotlib.pyplot as plt
import os

def plotting(filename, filename2=None, figsize=(4,4), capture=1, plot_comparison=False, plot_derivatives = False,
              plot_losses=False, plot_losses2=False, plot_lambdas=False, plot_losses_comparison=False, 
              plot_lambdas_comparison=False, plot_differences=False):
   
    W = np.load(os.path.join(filename, "W.npy"))
    dW_dx = np.load(os.path.join(filename, "dW_dx.npy"))
    dW_dxx = np.load(os.path.join(filename, "dW_dxx.npy"))
    dW_dxxx = np.load(os.path.join(filename, "dW_dxxx.npy"))
    dW_dxxxx = np.load(os.path.join(filename, "dW_dxxxx.npy"))
    x = np.load(os.path.join(filename, "x.npy"))
    w_analytic = np.load(os.path.join(filename, "w_analytic.npy"))
    lambdas = np.load(os.path.join(filename, "lambdas.npy"))
    losses = np.load(os.path.join(filename, "losses.npy"))

    epochs = capture*np.linspace(0,len(losses[:,0])-1, len(losses[:,0]))

    if (filename2 != None):
        W1 = np.load(os.path.join(filename2, "W.npy"))
        x1 = np.load(os.path.join(filename2, "x.npy"))
        w_analytic1 = np.load(os.path.join(filename2, "w_analytic.npy"))
        lambdas1 = np.load(os.path.join(filename2, "lambdas.npy"))
        losses1 = np.load(os.path.join(filename2, "losses.npy"))


    if (plot_comparison):
        plt.figure(figsize=figsize)
        plt.plot(x,w_analytic, '-', label='Analytical',  color='r')
        plt.plot(x,W, label='Predicted')
        plt.legend()
        plt.xlabel('Beam length / m')
        plt.ylabel('W(x) [m]')
        mape = np.mean(np.abs((w_analytic - W) / (w_analytic + 1e-30))) * 100 
        plt.title(f'Comparison of Analytical vs Predicted (MAPE: {mape:.2e}%)')
        plt.grid()
        plt.show()

    if (plot_derivatives):
        fig, axes = plt.subplots(4, 1, figsize=(figsize[0], 2*figsize[1]))
        axes[0].plot(x, dW_dx, label='dW_dx')
        axes[0].grid(True)
        axes[0].legend()
        axes[1].plot(x, dW_dxx, label='d^2W_dx^2')
        axes[1].grid(True)
        axes[1].legend()
        axes[2].plot(x, dW_dxxx, label='d^3W_dx^3')
        axes[2].grid(True)
        axes[2].legend()
        axes[3].plot(x, dW_dxxxx, label='d^4W_dx^4')
        axes[3].grid(True)
        axes[3].legend()        
        plt.xlabel('Beam length / m')
        plt.show()        

    if (plot_losses):
        plt.figure(figsize=figsize)
        plt.plot(epochs, lambdas[:,0]*losses[:,0], '-', label='Loss 0 (Physics)', color='k')
        plt.plot(epochs, lambdas[:,1]*losses[:,1], '-', label='Loss_1 (BC)', color='r')
        plt.plot(epochs, lambdas[:,2]*losses[:,2], '-', label='Loss_2 (BC)', color='b')
        plt.plot(epochs, lambdas[:,3]*losses[:,3], '-', label='Loss_3 (BC)', color='g')
        plt.plot(epochs, lambdas[:,4]*losses[:,4], '-', label='Loss_4 (BC)', color='orange')
        # plot the total loss multiplied by its corresponding lambda
        plt.plot(epochs, lambdas[:,0]*losses[:,0] + lambdas[:,1]*losses[:,1] + lambdas[:,2]*losses[:,2] + lambdas[:,3]*losses[:,3] + lambdas[:,4]*losses[:,4], '-', label='Total Loss', color='purple')
        L_average = np.mean(lambdas[:,0:5]*losses[:,0:5], axis=1)
        plt.plot(epochs, L_average, '--', label='Average Loss', color='purple')
        plt.yscale('log')
        # plt.ylim([10**(-14) ,10**(-3)])
        # plt.xlim([0 ,500])
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Loss-Value')
        plt.title('Losses')
        plt.show()

    if (plot_losses2):
        plt.figure(figsize=figsize)
        plt.plot(epochs, lambdas1[:,0]*losses1[:,0], '-', label='Loss 0 (Physics)', color='k')
        plt.plot(epochs, lambdas1[:,1]*losses1[:,1], '-', label='Loss_1 (BC)', color='r')
        plt.plot(epochs, lambdas1[:,2]*losses1[:,2], '-', label='Loss_2 (BC)', color='b')
        plt.plot(epochs, lambdas1[:,3]*losses1[:,3], '-', label='Loss_3 (BC)', color='g')
        plt.plot(epochs, lambdas1[:,4]*losses1[:,4], '-', label='Loss_4 (BC)', color='orange')
        # plot the total loss multiplied by its corresponding lambda
        plt.plot(epochs, lambdas1[:,0]*losses1[:,0] + lambdas1[:,1]*losses1[:,1] + lambdas1[:,2]*losses1[:,2] + lambdas1[:,3]*losses1[:,3] + lambdas1[:,4]*losses1[:,4], '-', label='Total Loss', color='purple')
        L_average = np.mean(lambdas1[:,0:5]*losses1[:,0:5], axis=1)
        plt.plot(epochs, L_average, '--', label='Average Loss', color='purple')
        plt.yscale('log')
        #plt.ylim([10**(-14) ,10**(-3)])
        #plt.xlim([0 ,500])
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Loss-Value')
        plt.title('Losses 2')
        plt.show()

    if (plot_lambdas):
        plt.figure(figsize=figsize)
        plt.plot(epochs, lambdas[:,0], '-', label='Lambda_0 (Physics)', color='k', linewidth='1')
        plt.plot(epochs, lambdas[:,1], '-', label='Lambda_1 (BC)', color='r', linewidth='1')
        plt.plot(epochs, lambdas[:,2], '-', label='Lambda_2 (BC)', color='b', linewidth='1')
        plt.plot(epochs, lambdas[:,3], '-', label='Lambda_3 (BC)', color='g', linewidth='1')
        plt.plot(epochs, lambdas[:,4], '-', label='Lambda_4 (BC)', color='orange', linewidth='1')
        plt.yscale('log')
        #plt.ylim([10**(-14) ,10**(-3)])
        #plt.xlim([0 ,500])
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Lambda-Value')
        plt.title('Lambdas')
        plt.show()

    if (plot_losses_comparison):
        plt.figure(figsize=figsize)
        plt.plot(epochs, abs(lambdas[:,0]*losses[:,0]-lambdas1[:,0]*losses1[:,0]), '-', label='Delta_Loss 0 (Physics)', color='k')
        plt.plot(epochs, abs(lambdas[:,1]*losses[:,1]-lambdas1[:,1]*losses1[:,1]), '-', label='Delta_Loss_1 (BC)', color='r')
        plt.plot(epochs, abs(lambdas[:,2]*losses[:,2]-lambdas1[:,2]*losses1[:,2]), '-', label='Delta_Loss_2 (BC)', color='b')
        plt.plot(epochs, abs(lambdas[:,3]*losses[:,3]-lambdas1[:,3]*losses1[:,3]), '-', label='Delta_Loss_3 (BC)', color='g')
        plt.plot(epochs, abs(lambdas[:,4]*losses[:,4]-lambdas1[:,4]*losses1[:,4]), '-', label='Delta_Loss_4 (BC)', color='orange')
        plt.xlabel('epochs')
        plt.grid()
        plt.legend()
        plt.title('Losses difference')
        plt.yscale('log')
        plt.show()
    
    if (plot_lambdas_comparison):
        plt.figure(figsize=figsize)
        plt.plot(epochs, abs(lambdas[:,0]-lambdas1[:,0]), '-', label='Delta_Lambda 0 (Physics)', color='k')
        plt.plot(epochs, abs(lambdas[:,1]-lambdas1[:,1]), '-', label='Delta_Lambda_1 (BC)', color='r')
        plt.plot(epochs, abs(lambdas[:,2]-lambdas1[:,2]), '-', label='Delta_Lambda_2 (BC)', color='b')
        plt.plot(epochs, abs(lambdas[:,3]-lambdas1[:,3]), '-', label='Delta_Lambda_3 (BC)', color='g')
        plt.plot(epochs, abs(lambdas[:,4]-lambdas1[:,4]), '-', label='Delta_Lambda_4 (BC)', color='orange')
        plt.xlabel('epochs')
        plt.grid()
        plt.legend()
        plt.title('Lambdas difference')
        # plt.yscale('log')
        plt.show()

    if (plot_differences):
        total_loss = lambdas[:,0]*losses[:,0] + lambdas[:,1]*losses[:,1] + lambdas[:,2]*losses[:,2] + lambdas[:,3]*losses[:,3] + lambdas[:,4]*losses[:,4]
        loss_diff = np.diff(total_loss)
        plt.figure(figsize=figsize)
        # plt.plot(epochs[1:], np.diff(total_loss), 'o-', label='Loss differenca along tima', color='k')
        plt.plot(epochs[1:], abs(loss_diff), 'o-', label='Loss difference along time', color='k')
        plt.xlabel('epochs')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.show()

def save_results(save_dir, W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, lambdas, losses):
    result_dir = os.path.join(os.getcwd(), "Outputs", save_dir)
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, "W.npy"), W.numpy()) 
    np.save(os.path.join(result_dir, "dW_dx.npy"), dW_dx.numpy())
    np.save(os.path.join(result_dir, "dW_dxx.npy"), dW_dxx.numpy())
    np.save(os.path.join(result_dir, "dW_dxxx.npy"), dW_dxxx.numpy())
    np.save(os.path.join(result_dir, "dW_dxxxx.npy"), dW_dxxxx.numpy())    
    np.save(os.path.join(result_dir, "x.npy"), x.numpy())
    np.save(os.path.join(result_dir, "w_analytic.npy"), w_analytic.numpy())
    np.save(os.path.join(result_dir, "lambdas.npy"), lambdas) 
    np.save(os.path.join(result_dir, "losses.npy"), losses)

def load_results(save_dir):
    result_dir = os.path.join(os.getcwd(), "Outputs", save_dir)
    W = np.load(os.path.join(result_dir, "W.npy"))
    dW_dx = np.load(os.path.join(result_dir, "dW_dx.npy"))
    dW_dxx = np.load(os.path.join(result_dir, "dW_dxx.npy"))
    dW_dxxx = np.load(os.path.join(result_dir, "dW_dxxx.npy"))
    dW_dxxxx = np.load(os.path.join(result_dir, "dW_dxxxx.npy"))
    x = np.load(os.path.join(result_dir, "x.npy"))
    w_analytic = np.load(os.path.join(result_dir, "w_analytic.npy"))
    lambdas = np.load(os.path.join(result_dir, "lambdas.npy"))
    losses = np.load(os.path.join(result_dir, "losses.npy"))
    return W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w_analytic, lambdas, losses