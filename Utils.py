import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

# Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device( 'cpu')

# PINN model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"tanh{i}", nn.Tanh())

    def forward(self, x):
        return self.net(x)

class Utils:

    def __init__(self, epochs=30001, inverse=False, scheme='own', conditioning=False):
        self.q_0 = 0.015
        self.l = 10
        self.EI = 20.83
        self.scheme = scheme
        self.epochs = epochs
        self.inverse = inverse
        self.conditioning = conditioning


    # Right-hand side of Bernoulli equation
    def f(self, x):
        return self.q_0*torch.sin(np.pi*x/self.l)

    # PDE residual
    def pde_loss(self, X):
        X.requires_grad_(True)
        u = self.model(X)
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, X, torch.ones_like(u_x), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, X, torch.ones_like(u_x), create_graph=True)[0]
        source = self.f(X)
        return torch.mean((source-u_xxxx*self.EI)**2)

    # Boundary loss
    def boundary_loss(self, Xb):
        Xb.requires_grad_(True)
        u=self.model(Xb)
        u_x = torch.autograd.grad(u, Xb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, Xb, torch.ones_like(u_x), create_graph=True)[0]
        loss_D = torch.mean(u**2)
        loss_N = torch.mean(u_xx**2)
        return loss_D, loss_N

    # Generate collocation and boundary points
    def generate_points(self, n_interior):
        x_in = torch.linspace(0,10,n_interior).reshape(n_interior,1).to(device)
        x_b = torch.tensor([[0],[10]], dtype=torch.float32).to(device)
        return x_in, x_b
    
    def balancing_scheme(self, loss_in, loss_bd, loss_bn):
        """ lam[0] -> pde loss
            lam[1] -> bd_loss
            lam[3] -> bn_loss
        """
        losses = [loss_in.detach().numpy(), loss_bd.detach().numpy(), loss_bn.detach().numpy()]
        # losses = np.array([loss_in, loss_bd, loss_bn], dtype=float)
        ratio = losses / min(losses)
        order = np.floor(np.log10(ratio))
        lam = 10**order

        # Accumulation
        # lam_in = lam[0]*lam_last[0]
        # lam_bd = lam[1]*lam_last[1]
        # lam_bn = lam[2]*lam_last[2]


        # No accumulation without conditioning
        lam_in = lam[0]
        lam_bd = lam[1]
        lam_bn = lam[2]

        if self.conditioning:
            # No accumulation with conditioning
            if max(lam_in, lam_bd, lam_bn) == lam_in:
                if lam_bn>=lam_bd:
                    return lam_in, lam_bd, lam_bn
                else:
                    lam_in = lam_bn
            else:
                if max(lam_in, lam_bd, lam_bn) == lam_bn:
                    lam_in = lam_bn
                elif max(lam_in, lam_bd, lam_bn) == lam_bd:
                    lam_in = lam_bd
                    lam_bn = lam_bd

        return lam_in, lam_bd, lam_bn

    def train(self):
        torch.manual_seed(42)
        layers = [1, 120, 120,  1]
        self.model = PINN(layers).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        X_in, X_b = self.generate_points(1000)
        self.loss_vec = []
        self.lambs_history = []
        self.epoch_triggered = []
        lam_in = lam_bd = lam_bn = 1
        patience   = 4        # epochs without improvement before triggering
        min_delta  = 0.0        # require at least this absolute improvement
        bad_epochs = 0
        best_loss  = float("inf")

        best_model = {
            "score": float("inf"),
            "state_dict": None,
            "optim_state": None,
            "epoch": -1,
            "triple": None,
        }

        L0_in = L0_bd = L0_bn = None  # Normalisierung
        print('Start training with epochs: ', self.epochs, ', scheme: ', self.scheme, ' and conditioning: ', self.conditioning)
        for epoch in range(self.epochs):

            optimizer.zero_grad()
            loss_in = self.pde_loss(X_in)
            loss_bd, loss_bn = self.boundary_loss(X_b)
            loss = (lam_in)* loss_in + (lam_bd)* loss_bd + (lam_bn)* loss_bn
            loss.backward()
            optimizer.step()
            self.loss_vec.append([loss.item(), loss_in.item(), loss_bd.item(), loss_bn.item()])
            self.lambs_history.append([lam_in, lam_bd, lam_bn])

            if epoch % 100 == 0:

                li  = float(loss_in.detach().cpu().item())
                lbd = float(loss_bd.detach().cpu().item())
                lbn = float(loss_bn.detach().cpu().item())

                if L0_in is None:
                    L0_in, L0_bd, L0_bn = max(li, 1e-30), max(lbd, 1e-30), max(lbn, 1e-30)
                norm_score = li/L0_in + lbd/L0_bd + lbn/L0_bn
                if norm_score < best_model["score"] - 1e-12:
                    best_model.update({
                        "score": norm_score,
                        "state_dict": copy.deepcopy(self.model.state_dict()),
                        "optim_state": copy.deepcopy(optimizer.state_dict()),
                        "epoch": epoch,
                        "triple": (li, lbd, lbn),
                    })
                    epoch_best_nodel = epoch
                loss_val = float(loss.detach().cpu().item())
                if loss_val < best_loss - min_delta:
                    best_loss = loss_val
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if (bad_epochs >= patience):
                    if self.scheme == 'own':
                        lam_in, lam_bd, lam_bn = self.balancing_scheme(loss_in, loss_bd, loss_bn)
                    print('Patience triggered, λ_in=',lam_in,', λ_bd=',lam_bd,', λ_bn=',lam_bn)
                    self.epoch_triggered.append(epoch)
                    bad_epochs = 0
                    best_loss  = float("inf")
                print(f"Epoch {epoch:05d}, L_in: {loss_in.item():.2e} | L_bd: {loss_bd.item():.2e} | L_bn: {loss_bn.item():.2e} | L_tot: {loss.item():.2e}")

        print('Last update of model at epoch: ', epoch_best_nodel, '\n')


    def plot_save_results(self, path, filename):
        self.model.cpu()
        x = torch.linspace(0,10,100).reshape(100,1)
        u_exact = self.q_0*self.l**4/(np.pi**4*self.EI)*np.sin(np.pi*x/self.l)
        u_pred = self.model(x).detach().numpy()
        u_exact = np.array(u_exact, dtype=float)
        u_exact[-1] = 0
        u_pred = np.array(u_pred, dtype=float)
        mask = u_exact != 0
        mape = np.mean(np.abs((u_exact[mask] - u_pred[mask]) / u_exact[mask])) * 100


        fig, ax = plt.subplots(1, 3, figsize=(18, 5))


        ax[0].plot(x,u_exact, label='exact')
        ax[0].plot(x,u_pred, label='pred')
        ax[0].set_title(f'mape: {mape:.2f}%')
        ax[0].legend()


        self.loss_vec = np.array(self.loss_vec)
        loss_to_hist = self.loss_vec[:,0]
        loss_in_hist = self.loss_vec[:,1]
        loss_bd_hist = self.loss_vec[:,2]
        loss_bn_hist = self.loss_vec[:,3]
        # ax[1].plot(loss_to_hist, label='loss_tot')
        ax[1].plot(loss_in_hist, label='loss_pde')
        ax[1].plot(loss_bd_hist, label='loss_bd')
        ax[1].plot(loss_bn_hist, label='loss_bn')
        ax[1].vlines(self.epoch_triggered, ymin=min(min(loss_in_hist), min(loss_bd_hist), min(loss_bn_hist)), ymax=1, linestyle='dashed', colors='gray')
        ax[1].set_yscale('log')
        ax[1].legend()

        lambs_history  = np.array(self.lambs_history)
        lam_in_histoty = lambs_history[:,0]
        lam_bd_histoty = lambs_history[:,1]
        lam_bn_histoty = lambs_history[:,2]
        ax[2].plot(lam_in_histoty, label='lam_pde')
        ax[2].plot(lam_bd_histoty, label='lam_bd')
        ax[2].plot(lam_bn_histoty, label='lam_bn')
        if self.scheme == 'own' :
            ax[2].set_yscale('log')
        ax[2].legend()

        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, filename)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)


