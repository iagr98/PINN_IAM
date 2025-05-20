import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from time import time
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda

class Utils:

    def __init__(self, F_val, L_val, EIz_val, num_points=300, n_data=2, num_collocation_points=300, normalize=True):
        self.F_val = F_val
        self.L_val = L_val
        self.EIz_val = EIz_val
        self.num_points = num_points
        self.n_data = n_data
        self.num_collocation_points = num_collocation_points
        self.normalize = normalize
        self.TOL = 1e-5
        self.inverse = False
        self.inverse_var = None
        self.num_b_losses = 4
        self.data_min = 0
        self.data_max = self.L_val
        self.current_losses = np.zeros((self.num_b_losses+1, 1))
        self.rng = np.random.RandomState(seed=42)
        self.is_plateau_tf = tf.Variable(False, trainable=False, dtype=tf.bool)


    def analytical_solution(self):
        x = sp.symbols('x')
        u_specific = (1/(self.EIz_val)) * (-((1/6)*self.F_val*x**3)+(0.5*self.F_val*self.L_val*x**2))
        
        u_numeric = sp.lambdify(x, u_specific)
        self.x_vals = np.linspace(0, self.L_val, self.num_points)
        self.u_vals = u_numeric(self.x_vals)

        return self.x_vals, self.u_vals
    
    def generate_data(self, plot):
        # Scalers for inptus (x) and outputs (u)
        self.scaler_in = MinMaxScaler()
        self.scaler_out = MinMaxScaler()

        # Data generation
        if (self.n_data ==2):
            self.x, self.u = self.analytical_solution()
            self.x_data = np.array([self.x[0], self.x[-1]])
            self.u_data = np.array([self.u[0], self.u[-1]])
        else:
            if (self.n_data!=1):
                self.num_points = self.num_points + 1; self.n_data = self.n_data -1
            else: None
            self.x, self.u = self.analytical_solution()
            self.x_data = self.x[0:self.num_points:int((self.num_points)/self.n_data)]
            self.u_data = self.u[0:self.num_points:int((self.num_points)/self.n_data)]
        self.x_physics = np.linspace(0, self.L_val, self.num_collocation_points)

        if (self.normalize):
            # Normalization of input and output data
            self.x_n = self.scaler_in.fit_transform(self.x.reshape(-1,1))
            self.u_n = self.scaler_out.fit_transform(self.u.reshape(-1,1))
            self.x_data_n = self.scaler_in.fit_transform(self.x_data.reshape(-1,1))
            self.u_data_n = self.scaler_out.fit_transform(self.u_data.reshape(-1,1))
            self.x_physics_n = self.scaler_in.fit_transform(self.x_physics.reshape(-1,1))
            self.L_val = 1.0
        else:
            self.x_n = self.x
            self.u_n = self.u
            self.x_data_n = self.x_data
            self.u_data_n = self.u_data
            self.x_physics_n = self.x_physics

            

        # ---------------------------------------------------- Plotting ----------------------------------------------------
        if (plot):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 Zeile, 2 Spalten

            # Plot 1: Normalized data
            axes[0].plot(self.x_n, self.u_n, label='Normalized analytical solution (self. Vs u_n)')
            axes[0].scatter(self.x_data_n, self.u_data_n, label='Normalized training dself.ata (x_data_n Vs u_data_n)', color="tab:orange")
            axes[0].scatter(self.x_physics_n, -0.1*np.ones(len(self.x_physics_n)), label='Collocation points (x_physics_n)', color='g')
            axes[0].legend()
            axes[0].set_xlabel('Length of the beam (Normalized)')
            axes[0].set_ylabel('Deflection (Normalized)')
            axes[0].set_title("Normalized Beam Data")

            # Plot 2: Unnormalized data
            axes[1].plot(self.x, self.u, label='Unnormalized analytical solution')
            axes[1].scatter(self.x_data, self.u_data, label='Unnormalized training data', color="tab:orange")
            axes[1].legend()
            axes[1].set_xlabel('Length of the beam')
            axes[1].set_ylabel('Deflection')
            axes[1].set_title("Unnormalized Beam Data")

            plt.tight_layout()
            plt.show()

    

    class NET(nn.Module):
        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation=nn.Tanh):
            super().__init__()
            activation = activation
            # Definition of input layer
            self.fcs = nn.Sequential(*[
                            nn.Linear(N_INPUT, N_HIDDEN),
                            activation()])
            # Definition of hidden layers
            self.fch = nn.Sequential(*[
                            nn.Sequential(*[
                                nn.Linear(N_HIDDEN, N_HIDDEN),
                                activation()]) for _ in range(N_LAYERS-1)])
            # Definition of output layer
            self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
            
        def forward(self, x):     # This function is to define how the data is going to me read during forward pass 
            x = self.fcs(x)       # 1. First the input data is read by neurons in the input layer
            x = self.fch(x)       # 2. The the data caming from input layers is spreaded and read by hidden layers
            x = self.fce(x)       # 3. Finally, after all data was spreaded along the hidden layers, it is passed to the output layer
            return x
        
    

    def train_DNN(self, model, epochs, lr=1e-3, x_data_n=[], u_data_n=[]):
        # Normalize data points of analytical solution (Training data)
        x_data_n = torch.tensor(self.x_data_n, dtype=torch.float32).view(-1, 1)
        u_data_n = torch.tensor(self.u_data_n, dtype=torch.float32).view(-1, 1)
        
        # Define the neural network
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training Loop
        tol = 1e-18
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            loss = self.data_loss(model, x_data_n, u_data_n)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 1000 == 0:
                print(f"Pretraining Epoch {epoch}, Loss: {loss.item():.6f}")
            
            if (loss < tol):
                print(f"Training terminated due to minimum tolerance {tol:e} was achieved")
                break
        
        print("Pretraining finished.")

        u_pred_n_DNN = model(torch.tensor(self.x_n, dtype=torch.float32).view(-1, 1)).detach().numpy()
        mse_DNN = np.mean((u_pred_n_DNN-self.u_n)**2)

        plt.figure(figsize=(8,3))
        plt.plot(self.x_n, u_pred_n_DNN, '--', label='Prediction made with DNN', linewidth=2, color='tab:orange')
        plt.plot(self.x_n, self.u_n, label='Normalized analytical solution (x_n Vs u_n)')
        plt.scatter(x_data_n, u_data_n, label='Normalized training data (x_data_n Vs u_data_n)', color="tab:orange")
        plt.title(f'(MSE={mse_DNN:e})')
        plt.legend()
        plt.show()
        
        return model
    
    def boundary_loss(self, model):
        x0 = torch.tensor([[0.0]], requires_grad=True)  # x = 0
        xL = torch.tensor([[float(self.L_val)]], requires_grad=True)   # x = L
        # Check requires_grad
        assert x0.requires_grad, "x0 does not require grad"
        assert xL.requires_grad, "xL does not require grad"
        assert model(x0).requires_grad, "model(x0) does not require grad"
        assert model(xL).requires_grad, "model(x0) does not require grad"
        
        # Boundary conditions at x = 0
        u_x0 = torch.autograd.grad(model(x0), x0, grad_outputs=torch.ones_like(model(x0)), create_graph=True)[0]
        u_xx_0 = torch.autograd.grad(u_x0, x0, grad_outputs=torch.ones_like(u_x0), create_graph=True)[0]
        loss_bc_1 = (model(x0) - 0)**2  # u(0) = 0
        loss_bc_2 = (u_x0 - 0)**2  # u_y'(0) = 0
        loss_bc_3 = (u_xx_0 - (-self.F_val * self.L_val / self.EIz_val))**2  # u_y''(0) = -FL / EI_z

        # Boundary conditions at x = L
        u_x_L = torch.autograd.grad(model(xL), xL, grad_outputs=torch.ones_like(model(xL)), create_graph=True)[0]
        u_xx_L = torch.autograd.grad(u_x_L, xL, grad_outputs=torch.ones_like(u_x_L), create_graph=True)[0]
        u_xxx_L = torch.autograd.grad(u_xx_L, xL, grad_outputs=torch.ones_like(u_xx_L), create_graph=True)[0]
        loss_bc_4 = (u_xx_L)**2  # u''(L) = 0
        loss_bc_5 = (u_xxx_L + self.F_val / self.EIz_val)**2  # u'''(L) = -F/EIz
        
        return loss_bc_1, loss_bc_2, loss_bc_3, loss_bc_4, loss_bc_5

    # Data loss function
    def data_loss(self,model, x_data, u_data):
        u_pred = model(x_data)
        loss_data = torch.mean((u_pred - u_data)**2)  # Mean squared error
        return loss_data

    # Physics-Informed Loss
    def physics_loss(self,model, x):
        x.requires_grad = True
        u_pred = model(x)
        u_xx = torch.autograd.grad(torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0],x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0], x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
        # Check requires_grad
        assert u_pred.requires_grad, "u_pred does not require grad"
        assert u_xx.requires_grad, "u_xx does not require grad"
        assert u_xxxx.requires_grad, "u_xxxx does not require grad"
        
        #residual = EIz * u_xxxx  # EI_z * d^4u/dx^4 = 0
        residual = u_xxxx  # d^4u/dx^4 = 0
        return torch.mean(residual**2)
    

    def train_PINN(self, model, epochs, w_bc_1, w_bc_2, w_bc_3, w_bc_4, w_data, w_pde, T, alpha, rho, lr=1e-3, validation_split=0.2):
        # Normalize data points of analytical solution (Training data)
        x_data_n = torch.tensor(self.x_data_n, dtype=torch.float32).view(-1, 1)
        u_data_n = torch.tensor(self.u_data_n, dtype=torch.float32).view(-1, 1)

        # Split data into training and validation sets
        val_size = int(validation_split * len(self.x_data_n))
        train_size = len(self.x_data_n) - val_size

        _, x_val = torch.split(x_data_n, [train_size, val_size])
        _, u_val = torch.split(u_data_n, [train_size, val_size])
        
        # Physics colocation points
        x_physics = torch.tensor(self.x_physics_n, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        
        # Define optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Initialize ReLoBRaLo loss tracking
        loss_history = {i: torch.tensor(1.0, dtype=torch.float32) for i in range(6)}  # 6 loss components
        loss_initial = {i: torch.tensor(1.0, dtype=torch.float32) for i in range(6)}  # Store initial loss values
        lambs_prev = {i: torch.tensor(1.0, dtype=torch.float32) for i in range(6)}  # Previous lambda values

        
        # Save forward pass prediction (Prediction before training)
        self.u_fp_n = model(x_physics)
        
        # Lists to store loss history
        self.train_losses, val_losses = [], []
        
        # Training Loop
        tol = 1e-8
        start_time = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute losses
            loss_bc_1, loss_bc_2, loss_bc_3, loss_bc_4, loss_bc_5 = self.boundary_loss(model)
            loss_data = self.data_loss(model, x_data_n, u_data_n).clone().detach().requires_grad_(True).view(1, 1)
            #loss_data = torch.tensor(0.0, dtype=torch.float32).view(-1, 1)
            loss_pde = self.physics_loss(model, x_physics).clone().detach().requires_grad_(True).view(1, 1)

            # Store losses in a list
            losses = [loss_bc_1, loss_bc_2, loss_bc_3, loss_bc_4, loss_bc_5, loss_data, loss_pde]
            loss_tensor = torch.stack(losses)

            # Compute loss ratios for ReLoBRaLo
            loss_ratios = loss_tensor / (torch.tensor([loss_history[i] for i in range(6)], dtype=torch.float32) * T + 1e-12)
            loss_initial_ratios = loss_tensor / (torch.tensor([loss_initial[i] for i in range(6)], dtype=torch.float32) * T + 1e-12)

            # Compute softmax-based scaling factors
            lambs_hat = F.softmax(loss_ratios, dim=0) * len(losses)
            lambs0_hat = F.softmax(loss_initial_ratios, dim=0) * len(losses)

            # Apply ReLoBRaLo update with rho
            lambs = [
                rho * alpha * lambs_prev[i] + (1 - rho) * alpha * lambs0_hat[i] + (1 - alpha) * lambs_hat[i]
                for i in range(6)
            ]

            print(lambs)
        
            # Update loss history and previous lambda values
            for i in range(6):
                loss_history[i] = losses[i].detach()
                lambs_prev[i] = lambs[i].detach()

            # Compute weighted loss
            loss_train = lambs[0][0,0]*losses[0] + lambs[1][0,0]*losses[1] + lambs[2][0,0]*losses[2] + lambs[3][0,0]*losses[3] + lambs[4][0,0]*losses[4] + lambs[5][0,0]*losses[5]
            #loss_train = sum(lambs[i] * losses[i] for i in range(6))
            loss_train.backward()
            optimizer.step()

            
            # Total loss
            loss_bc = loss_bc_1 + loss_bc_2 + loss_bc_3 + loss_bc_4 + loss_bc_5
            #loss_train = w_bc_1*loss_bc_1 + w_bc_2*loss_bc_2 + w_bc_3*loss_bc_3 + w_bc_4*loss_bc_4 + w_data*loss_data + w_pde*loss_pde
            #loss_train.backward()
            #optimizer.step()

            with torch.no_grad():
                if (x_val.numel() > 0) :   # Compute validation loss if x_val is not empty (x_val != [])
                    val_loss = self.data_loss(model, x_val, u_val)  # Compute validation loss
                else: 
                    val_loss = torch.tensor([0.0])
            
            # Print progress
            if epoch % 10 == 0:
                
                self.train_losses.append(loss_train.sum().item())
                print(f"Epoch {epoch}, Total Loss: {loss_train.sum().item():e}, "
                    f"Validation Loss: {val_loss.item():e}, "
                    f"Boundary Loss: {(loss_bc).item():e}, "
                    f"Data Loss: {(w_data*loss_data).item():e}, "
                    f"Physics Loss: {(w_pde*loss_pde).item():e}")
                
            if (loss_train.sum() < tol):
                print(f"Training terminated due to minimum tolerance {tol:e} was achieved")
                break

            
        
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Runtime: {computation_time:.6f} [s]")
        return model, self.u_fp_n, self.train_losses
    



    def plot_post_training(self, model):
        u_pred_n = model(torch.tensor(self.x_n, dtype=torch.float32).view(-1, 1)).detach().numpy()
        mse = np.mean((u_pred_n-self.u_n)**2)

        num_epochs = np.linspace(0,len(self.train_losses), len(self.train_losses))

        plt.figure(figsize=(5, 5))

        #plt.subplot(1, 2, 1)
        plt.plot(self.x_n, self.u_n, label='n-Analytical Solution', linewidth=2)
        plt.plot(self.x_n, u_pred_n, '--', label='n-PINN Prediction after training', linewidth=2, color='tab:orange')
        plt.plot(self.x_physics_n, self.u_fp_n.detach().numpy(), '--', label='n-PINN Prediction before training', linewidth=2, color='g')
        plt.scatter(self.x_data_n, self.u_data_n, label='n-Training data used for DNN', color='tab:orange')
        plt.xlabel('n-x [m]')
        plt.ylabel('n-u_y(x)')
        plt.legend()
        #plt.title(f'Bernoulli Beam Deflection: n-Analytical solution vs n-PINN. (MSE={mse:e})')

        #plt.subplot(1, 2, 2)
        #plt.plot(num_epochs,self.train_losses, color='tab:blue', linewidth=2)
        #plt.yscale('log')
        #plt.xlabel('Epochs * 1000')
        #plt.ylabel('Loss')
        #plt.title('Training losses')

        plt.tight_layout()
        plt.show()



    #############################################################################################################################
    ###### Implementation using Tensorflow

    
    def fully_connected(self, nlayers, nnodes, activation=tf.nn.tanh, name='fully_connected'):
        x = Input((1,), name='x')
        u = (x - self.data_min) / (self.data_max - self.data_min) * 2 - 1
        kernel_init = tf.keras.initializers.GlorotNormal(seed=42)  # Fixed seed
        u = Dense(nnodes, activation=activation, kernel_initializer=kernel_init, name='dense0')(u)
        for i in range(1, nlayers):
            u = Dense(nnodes, activation=activation, kernel_initializer=kernel_init, name=f'dense{i}')(u) + u
        u = Dense(1, activation='sigmoid', kernel_initializer=kernel_init)(u)
        return Model(x, u, name=name)

    
    def training_batch(self, batch_size:int=1024):
        " Sample points along the length of the beam "
        #########
        # x = np.random.uniform(0, self.L_val, size=(batch_size, 1))
        x = self.rng.uniform(0, self.L_val, size=(batch_size, 1))
        #########
        zero_tensor = tf.constant([[0.0]], dtype=tf.float32)        # Shape (1, 1)
        lval_tensor = tf.constant([[self.L_val]], dtype=tf.float32) # Shape (1, 1)
        return tf.cast(tf.concat([zero_tensor, x, lval_tensor], axis=0), dtype=tf.float32)
    
    def validation_batch(self):
        x, w = self.analytical_solution()
        x = tf.cast(x.reshape(-1, 1), dtype=tf.float32)
        w = tf.cast(w.reshape(-1, 1), dtype=tf.float32)
        return x, w
    
    @tf.function
    def derivatives(self, model:tf.keras.Model, x, training:bool=False):
        W = model[0](x, training=training)
        dW_dx = tf.gradients(W, x)[0]
        dW_dxx = tf.gradients(dW_dx, x)[0]
        dW_dxxx = tf.gradients(dW_dxx, x)[0]
        dW_dxxxx = tf.gradients(dW_dxxx, x)[0]
        return W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx

    @tf.function
    def calculate_loss(self, model:tf.keras.Model, x, aggregate_boundaries:bool=False, training:bool=False):
        W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx = self.derivatives(model, x, training=training)
        
        def non_plateau():
            f_loss = tf.reduce_mean(dW_dxxxx**2)
            xl = tf.cast(x < self.TOL, dtype=tf.float32)
            xu = tf.cast(x > self.L_val - self.TOL, dtype=tf.float32)
            b1_loss = tf.reduce_mean((xl * W)**2)
            b2_loss = tf.reduce_mean((xu * (W - 1e-3))**2)
            b3_loss = tf.reduce_mean((xl * (dW_dxx - (-self.F_val * self.L_val / self.EIz_val)))**2)
            b4_loss = tf.reduce_mean((xu * dW_dxx)**2)
            return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss]

        def plateau():
            gamma = 0.5
            f_loss, [b1_loss, b2_loss, b3_loss, b4_loss] = non_plateau()
            L_average = tf.reduce_mean([f_loss, b1_loss, b2_loss, b3_loss, b4_loss])
            f_loss += gamma * (L_average - f_loss)
            b1_loss += gamma * (L_average - b1_loss)
            b2_loss += gamma * (L_average - b2_loss)
            b3_loss += gamma * (L_average - b3_loss)
            b4_loss += gamma * (L_average - b4_loss)
            return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss]

        return tf.cond(pred=tf.equal(self.is_plateau_tf, False),true_fn=non_plateau,false_fn=plateau)       
    
    @tf.function
    def validation_loss(self, model:tf.keras.Model, x, w):
        w_pred = model[0](x, training=False)
        return tf.reduce_mean((w - w_pred)**2)
    
    @tf.function
    def relobralo(self, model, x, args:dict):
        f_loss, b_losses = self.calculate_loss(model, x, aggregate_boundaries=False, training=True)

        T = args['T']
        losses = [f_loss] + b_losses

        lambs_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(args['l'+str(i)]*T+1e-12) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
        lambs0_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(args['l0'+str(i)]*T+1e-12) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
        lambs = [args['rho']*args['alpha']*args['lam'+str(i)] + (1-args['rho'])*args['alpha']*lambs0_hat[i] + (1-args['alpha'])*lambs_hat[i] for i in range(len(losses))]
        
        loss = tf.reduce_sum([lambs[i]*losses[i] for i in range(len(losses))])
        grads = [tf.gradients(loss, model[0].trainable_variables)]

        # update args
        args = args.copy()
        for i in range(len(b_losses)+1):
            args['lam'+str(i)] = lambs[i]
            args['l'+str(i)] = losses[i]
        return grads, f_loss, b_losses, args
    
    def train(self, nlayers=4, nnodes=360, lr=0.001, epochs=5001, batch_size=1024, resample=True, T=0.1, alpha=0.999, rho=1, patience=4, factor=0.1, capture=1, strategy=True):
        model = [self.fully_connected(nlayers, nnodes)]
        # print(model[0].layers[1].get_weights()[0][:5])  # First 5 weights of the first layer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        args = {"lam"+str(i): tf.constant(1.) for i in range(self.num_b_losses+1)}
        args.update({"l"+str(i): tf.constant(1.) for i in range(self.num_b_losses+1)})
        args.update({"l0"+str(i): tf.constant(1.) for i in range(self.num_b_losses+1)})
        args["T"] = tf.constant(T, dtype=tf.float32)
        # rho_schedule = (np.random.uniform(size=epochs+1) < rho).astype(int).astype(np.float32)
        rho_schedule = (self.rng.uniform(size=epochs+1) < rho).astype(int).astype(np.float32)
        args['rho'] = tf.constant(rho_schedule[0], dtype=tf.float32)
        alpha_schedule = [tf.constant(1., tf.float32), tf.constant(0., tf.float32)] + [tf.constant(alpha, tf.float32)]
        args["alpha"] = alpha_schedule[0]
        best_loss = 1e9
        cooldown = patience
        plateau_count = 0
        plateau_threshold = 5   # Number of consecutive epochs with minimal improvement to consider a plateau
        min_delta = 1e-13        # Minimum improvement to consider as progress

        print('Start training of PINN using Tensorflow')
        start = time()
        x = self.training_batch(batch_size)
        lambdas = []
        losses = []
        previous_loss = None
        

        for epoch in range(epochs):
            if resample:
                x = self.training_batch(batch_size)
            grads, f_loss, b_losses, args = self.relobralo(model, x, args)
            optimizer.apply_gradients(zip(grads[0], model[0].trainable_variables))
            self.current_losses = [args['l'+str(i)].numpy() for i in range(self.num_b_losses+1)]

            if (epoch == 1):
                for i in range(self.num_b_losses+1):
                    args['l0'+str(i)] = ([f_loss]+b_losses)[i]
            if len(alpha_schedule) > 1:
                args['alpha'] = alpha_schedule[1]
                alpha_schedule = alpha_schedule[1:]
            args['rho'] = rho_schedule[1]
            rho_schedule = rho_schedule[1:]
            
            if epoch % capture == 0:
                x_val, w_val = self.validation_batch()
                val_loss = self.validation_loss(model, x_val, w_val)
                loss = f_loss + tf.reduce_sum(b_losses)
                print(
                    f"epoch {int(epoch)}: "
                    f"loss={loss.numpy():.3e}, "
                    f"val. loss={val_loss.numpy():.3e}, "
                    f"plateau={self.is_plateau_tf.numpy()}"
                )
                lambdas.append([args[f"lam{i}"].numpy() for i in range(len(b_losses)+1)])
                losses.append([args[f"l{i}"].numpy() for i in range(len(b_losses)+1)])

                ################################################################################################
                if (strategy):
                    if (previous_loss is not None):
                        improvement = abs(previous_loss - loss.numpy())
                        if (improvement < min_delta):
                            plateau_count += 1
                        else:
                            plateau_count = 0

                        if (plateau_count >= plateau_threshold):
                            self.is_plateau_tf.assign(True)
                            # plateau_count = 0
                        else:
                            self.is_plateau_tf.assign(False)
                        

                    previous_loss = loss.numpy()
                ################################################################################################

                if loss < best_loss:
                    best_loss = loss
                    cooldown = patience
                if cooldown <= 0 and epoch > epochs / 10:
                    cooldown = patience
                    optimizer.learning_rate.assign(optimizer.learning_rate * factor)
                cooldown -= 1
        end = time()
        print('Training completed. Elapsed time: ',(end-start),'s')
        
        x, w = self.validation_batch()
        W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx = self.derivatives(model, x, training=False)
        return W, dW_dx, dW_dxx, dW_dxxx, dW_dxxxx, x, w, model, np.array(lambdas), np.array(losses)
