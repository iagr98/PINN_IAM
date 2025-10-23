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
    def __init__(self, layers, add_inverse=False):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"tanh{i}", nn.Tanh())
        if add_inverse:
            self.EI_pred = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t):
        XT = torch.cat([x, t], dim=1)
        return self.net(XT)

class Utils:

    def __init__(self, epochs=30001, inverse=False, scheme='own', conditioning=False):
        self.q_0 = 0.015
        self.l = 10
        self.EI = 20.83
        self.mu = 1
        self.scheme = scheme
        self.epochs = epochs
        self.inverse = inverse
        self.conditioning = conditioning


    def calculate_loss(self, X_in, X_b, t_in, t_b):

        # Right-hand side of Bernoulli equation
        def f(x):
            return self.q_0*torch.sin(np.pi*x/self.l)
        
        def q(x,t):
            return self.q_0*torch.sin(np.pi*x/self.l)*torch.sin(np.pi*t)
        
        # Analytical solution
        def u_analytical(x):
            # x = x.detach().cpu().numpy()
            return self.q_0*self.l**4/(np.pi**4*self.EI)*torch.sin(np.pi*x/self.l)
        
        # PDE residual
        def pde_loss(X, t):
            X.requires_grad_(True)
            t.requires_grad_(True)
            u = self.model(X, t)
            u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0]
            u_xxx = torch.autograd.grad(u_xx, X, torch.ones_like(u_xx), create_graph=True)[0]
            u_xxxx = torch.autograd.grad(u_xxx, X, torch.ones_like(u_xxx), create_graph=True)[0]
            u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
            u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
            source = q(X,t)
            if self.inverse:
                return torch.mean((source/self.model.EI_pred-u_xxxx)**2)
            else:
                return torch.mean((source-u_xxxx*self.EI-u_tt*self.mu)**2)
            
        def boundary_loss(Xb, t):
                Xb.requires_grad_(True)
                u = self.model(Xb,t)
                u_x = torch.autograd.grad(u, Xb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x, Xb, torch.ones_like(u_x), create_graph=True)[0]
                loss_D = torch.mean(u**2)
                loss_N = torch.mean(u_xx**2)
                return loss_D, loss_N
        
        def initial_loss(X,t_0):
            u = self.model(X,t_0)
            return torch.mean(u**2)
        
        if self.inverse:
            loss_pde = pde_loss(X_in)
            u_exact = u_analytical(X_in)
            loss_u = torch.mean((u_exact - self.model(X_in))**2)
            return [loss_pde, loss_u]

        else:
            t_0 = torch.zeros_like(X_in)
            loss_in = pde_loss(X_in, t_in)
            loss_bd, loss_bn = boundary_loss(X_b, t_b)
            loss_ic = initial_loss(X_in, t_0)
            return [loss_in, loss_bd, loss_bn, loss_ic]


    # Generate collocation and boundary points
    def generate_points(self, n_interior, n_time):
        x_in = torch.linspace(0,10,n_interior).reshape(n_interior,1).to(device)
        x_b = torch.tensor([[0],[10]], dtype=torch.float32).to(device)
        t = torch.linspace(0,4,n_time).reshape(n_time,1).to(device)
        return x_in, x_b, t
    
    def balancing_scheme(self, losses):
        
        m = 2 if self.inverse else 3
        to_np = lambda t: float(t.detach().cpu())  # losses are scalars
        losses = np.array([to_np(losses[i]) for i in range(m)], dtype=np.float64)

        ratio = losses / min(losses)
        order = np.floor(np.log10(ratio))
        lam = 10**order

    
        if self.conditioning:

            if self.inverse:
                lam_pde = lam[0]
                lam_u = lam[1]
                if (lam_pde>=lam_u):
                    lam = [lam_pde, lam_u]
                else:
                    lam_pde = lam_u
                    lam = [lam_pde, lam_u]
                return lam
            else:

                lam_in = lam[0]
                lam_bd = lam[1]
                lam_bn = lam[2]

                # No accumulation with conditioning
                if max(lam_in, lam_bd, lam_bn) == lam_in:
                    if lam_bn>=lam_bd:
                        lam = [lam_in, lam_bd, lam_bn]
                        return lam
                    else:
                        lam_in = lam_bn
                else:
                    if max(lam_in, lam_bd, lam_bn) == lam_bn:
                        lam_in = lam_bn
                    elif max(lam_in, lam_bd, lam_bn) == lam_bd:
                        lam_in = lam_bd
                        lam_bn = lam_bd
                lam = [lam_in, lam_bd, lam_bn]

                return lam

    def relobralo(self, losses_cur, losses_old, losses_0, lambdas,
              alpha=0.5, rho=0.5, m=3, T=0.1, eps=1e-12):
        """
        Returns a 3-vector (lam_in_new, lam_bd_new, lam_bn_new).
        All 'loss_*' inputs are torch tensors (scalars). lambdas are floats.
        """

        # --- Convert tensors → float64 numpy scalars (safe for GPU tensors) ---
        to_np = lambda t: float(t.detach().cpu())  # losses are scalars
        m = 2 if self.inverse else 3
        losses_cur = np.array([to_np(losses_cur[i]) for i in range(m)], dtype=np.float64)
        losses_old = np.array([to_np(losses_old[i]) for i in range(m)], dtype=np.float64)
        losses_0   = np.array([to_np(losses_0[i]) for i in range(m)], dtype=np.float64)
        lambs_cur  = np.array([float(lambdas[i]) for i in range(m)], dtype=np.float64)

        # --- Component-wise ratios; softmax-style normalization with temperature T ---
        # Stabilize exp by subtracting max before exponentiation
        z_cur = losses_cur / (T * losses_old + eps)
        z_cur = z_cur - z_cur.max()
        e_cur = np.exp(z_cur)
        lam_bal_cur = m * e_cur / (e_cur.sum() + eps)   # 3-vector

        z_0 = losses_cur / (T * losses_0 + eps)
        z_0 = z_0 - z_0.max()
        e_0 = np.exp(z_0)
        lam_bal_0 = m * e_0 / (e_0.sum() + eps)         # 3-vector

        # --- Historical smoothing and final update (all 3-vectors) ---
        lambs_hist = rho * lambs_cur + (1.0 - rho) * lam_bal_0
        lambs_new  = alpha * lambs_hist + (1.0 - alpha) * lam_bal_cur

        # Return as tuple for easy unpacking: lam_in_new, lam_bd_new, lam_bn_new
        return tuple(lambs_new.tolist())

    def train(self):
        torch.manual_seed(42)
        layers = [2, 120, 120,  1]
        self.model = PINN(layers, add_inverse=self.inverse).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        X_in, X_b, t = self.generate_points(100, 40)
        # Build full (x,t) meshgrid for interior and boundary points
        X, T = torch.meshgrid(X_in.squeeze(), t.squeeze(), indexing='ij')
        X_in = X.reshape(-1, 1)
        t_in = T.reshape(-1, 1)
        X, T = torch.meshgrid(X_b.squeeze(), t.squeeze(), indexing='ij')
        X_b = X.reshape(-1, 1)
        t_b = T.reshape(-1, 1)
        self.loss_vec = []
        self.lambs_history = []
        self.epoch_triggered = []
        lambdas = [1, 1, 1, 1]
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
        if self.scheme=='relobralo':
            if self.inverse:
                losses = torch.tensor([1.0, 1.0], dtype=torch.float32)
            else:
                losses = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        L0_in = L0_bd = L0_bn = L0_ic = L0_pde = L0_u = None  # Normalisierung
        print('Start training with epochs:', self.epochs, ', scheme:', self.scheme, ', conditioning:', self.conditioning, ', Inverse:', self.inverse)
        for epoch in range(self.epochs):

            optimizer.zero_grad()
            if (self.scheme=='relobralo'):
                losses_old = losses
            losses = self.calculate_loss(X_in, X_b, t_in, t_b)
            # losses = [loss_in, loss_bd, loss_bn]
            if (self.scheme=='relobralo'):
                if (epoch==0):
                    losses_0 = losses
                lambdas = self.relobralo(losses, losses_old, losses_0, lambdas)
            if self.inverse:
                loss = (lambdas[0])* losses[0] + (lambdas[1])* losses[1]
            else:
                loss = (lambdas[0])* losses[0] + (lambdas[1])* losses[1] + (lambdas[2])* losses[2] + (lambdas[3])* losses[3]
            loss.backward()
            optimizer.step()
            if self.inverse:
                self.loss_vec.append([loss.item(), losses[0].item(), losses[1].item(), self.model.EI_pred.item()])
                self.lambs_history.append([lambdas[0], lambdas[1]])
            else:
                self.loss_vec.append([loss.item(), losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
                self.lambs_history.append([lambdas[0], lambdas[1], lambdas[2], lambdas[3]])

            if epoch % 10 == 0:

                if self.inverse:
                    l_pde  = float(losses[0].detach().cpu().item())
                    l_u = float(losses[1].detach().cpu().item())
                else:
                    li  = float(losses[0].detach().cpu().item())
                    lbd = float(losses[1].detach().cpu().item())
                    lbn = float(losses[2].detach().cpu().item())
                    lic = float(losses[3].detach().cpu().item())

                if (self.inverse==False and L0_in is None) or (self.inverse and L0_pde is None):
                    if self.inverse:
                        L0_pde, L0_u = max(l_pde, 1e-30), max(l_u, 1e-30)
                    else:
                        L0_in, L0_bd, L0_bn, L0_ic = max(li, 1e-30), max(lbd, 1e-30), max(lbn, 1e-30), max(lic, 1e-30)
                norm_score = l_pde/L0_pde + l_u/L0_u if self.inverse else li/L0_in + lbd/L0_bd + lbn/L0_bn + lic/L0_ic
                if norm_score < best_model["score"] - 1e-12:
                    best_model.update({
                        "score": norm_score,
                        "state_dict": copy.deepcopy(self.model.state_dict()),
                        "optim_state": copy.deepcopy(optimizer.state_dict()),
                        "epoch": epoch,
                        "triple": (l_pde, l_u) if self.inverse else (li, lbd, lbn),
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
                        lambdas = self.balancing_scheme(losses)
                        if self.inverse:
                            print('Patience triggered, λ_pde=',lambdas[0],', λ_u=',lambdas[1])
                        else:
                            print('Patience triggered, λ_in=',lambdas[0],', λ_bd=',lambdas[1],', λ_bn=',lambdas[2])
                    self.epoch_triggered.append(epoch)
                    bad_epochs = 0
                    best_loss  = float("inf")
                if (self.inverse):
                    print(f"Epoch {epoch:05d}, L_pde: {losses[0].item():.2e} | L_u: {losses[1].item():.2e} | L_tot: {loss.item():.2e} | EI_pred: {self.model.EI_pred.item()}")
                else:
                    print(f"Epoch {epoch:05d}, L_in: {losses[0].item():.2e} | L_bd: {losses[1].item():.2e} | L_bn: {losses[2].item():.2e} | L_ic: {losses[3].item():.2e} | L_tot: {loss.item():.2e}")

        print('Last update of model at epoch: ', epoch_best_nodel, '\n')


    def plot_save_results(self, path, filename):
        self.model.cpu()
        x = torch.linspace(0,10,100).reshape(100,1)
        t = torch.linspace(0,4,40).reshape(40,1)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
        x = X.reshape(-1, 1)
        t = T.reshape(-1, 1)
        u_pred = self.model(x,t).detach().numpy()
        u_pred = np.array(u_pred, dtype=float)
        U_pred = u_pred.reshape(X.shape)


        
        self.loss_vec = np.array(self.loss_vec)

        if self.inverse:
            fig, ax = plt.subplots(1, 4, figsize=(18, 5))

            ax[0].plot(x,u_exact, label='exact')
            ax[0].plot(x,u_pred, label='pred')
            ax[0].set_title(f'mape: {mape:.2f}%')
            ax[0].legend()

            ax[1].plot(self.loss_vec[:,3], label='EI_pred')
            ax[1].hlines(self.EI, xmin=0, xmax=len(self.loss_vec[:,0]), label='EI_exact',linestyle='dashed', colors='gray')
            ax[1].legend()

            loss_to_hist = self.loss_vec[:,0]
            loss_pde_hist = self.loss_vec[:,1]
            loss_u_hist = self.loss_vec[:,2]
            # ax[2].plot(loss_to_hist, label='loss_tot')
            ax[2].plot(loss_pde_hist, label='loss_pde')
            ax[2].plot(loss_u_hist, label='loss_u')
            ax[2].vlines(self.epoch_triggered, ymin=min(min(loss_pde_hist), min(loss_u_hist)), ymax=1, linestyle='dashed', colors='gray')
            ax[2].set_yscale('log')
            ax[2].legend()

            lambs_history  = np.array(self.lambs_history)
            lam_pde_histoty = lambs_history[:,0]
            lam_u_histoty = lambs_history[:,1]

            ax[3].plot(lam_pde_histoty, label='lam_pde')
            ax[3].plot(lam_u_histoty, label='lam_u')
            if self.scheme == 'own' :
                ax[3].set_yscale('log')
            ax[3].legend()

        else:
            fig, ax = plt.subplots(1, 2, figsize=(18, 5))

            # 3D plot for ax[0]
            ax[0].pcolormesh(X, T, U_pred, cmap='viridis', shading='auto')
            ax[0].set_title('Predicted Surface')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('t')
            cbar = fig.colorbar(ax[0].collections[0], ax=ax[0], orientation='vertical', pad=0.02)
            cbar.set_label('u_pred')

            loss_to_hist = self.loss_vec[:,0]
            loss_in_hist = self.loss_vec[:,1]
            loss_bd_hist = self.loss_vec[:,2]
            loss_bn_hist = self.loss_vec[:,3]
            loss_ic_hist = self.loss_vec[:,4]
            ax[1].plot(loss_to_hist, label='loss_tot')
            ax[1].plot(loss_in_hist, label='loss_pde')
            ax[1].plot(loss_bd_hist, label='loss_bd')
            ax[1].plot(loss_bn_hist, label='loss_bn')
            ax[1].plot(loss_ic_hist, label='loss_ic')
            ax[1].vlines(self.epoch_triggered, ymin=min(min(loss_in_hist), min(loss_bd_hist), min(loss_bn_hist)), ymax=1, linestyle='dashed', colors='gray')
            ax[1].set_yscale('log')
            ax[1].set_xlabel('Epochs')
            ax[1].legend()

            # lambs_history  = np.array(self.lambs_history)
            # lam_in_histoty = lambs_history[:,0]
            # lam_bd_histoty = lambs_history[:,1]
            # lam_bn_histoty = lambs_history[:,2]
            # ax[2].plot(lam_in_histoty, label='lam_pde')
            # ax[2].plot(lam_bd_histoty, label='lam_bd')
            # ax[2].plot(lam_bn_histoty, label='lam_bn')
            # if self.scheme == 'own' :
            #     ax[2].set_yscale('log')
            # ax[2].legend()

        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, filename)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)


