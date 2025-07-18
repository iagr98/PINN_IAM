import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda

class Utils:

    def __init__(self, L_val, EI_val, q_0):
        self.L_val = L_val
        self.EI_val = EI_val
        self.q_0 = q_0           
        self.TOL = 1e-5
        self.inverse = False
        self.inverse_var = None
        self.num_b_losses = 4
        self.data_min = 0
        self.data_max = self.L_val
        self.current_losses = np.zeros((self.num_b_losses+1, 1))
        self.is_plateau_tf = tf.Variable(False, trainable=False, dtype=tf.bool)


    def analytical_solution(self):
        x = sp.symbols('x')
        u_specific = (((self.q_0*self.L_val**4)/(np.pi**4*self.EI_val)) * sp.sin(np.pi*x/self.L_val))
        u_numeric = sp.lambdify(x, u_specific)
        self.x_vals = np.linspace(0, self.L_val, 300)
        self.u_vals = u_numeric(self.x_vals)

        return self.x_vals, self.u_vals
    
    def fully_connected(self, nlayers, nnodes, activation=tf.math.sin, name='fully_connected'):
        x = Input((1,), name='x')
        u = (x - self.data_min) / (self.data_max - self.data_min) * 2 - 1
        kernel_init = tf.keras.initializers.GlorotNormal(seed=200)  ########## Fixed seed for reproducibility
        u = Dense(nnodes, activation=activation, kernel_initializer=kernel_init, name='dense0')(u)
        for i in range(1, nlayers):
            u = Dense(nnodes, activation=activation, kernel_initializer=kernel_init, name='dense'+str(i))(u) + u
        u = Dense(1, activation='sigmoid', kernel_initializer=kernel_init)(u)
        return Model(x, u, name=name)

    
    def training_batch(self, batch_size:int=38, n_boundary_points=4):
        # " Sample points along the length of the beam "
        # np.random.seed(42)  ########## Fixed seed for reproducibility
        # x = np.random.uniform(0, self.L_val, size=(batch_size, 1))
        # zero_tensor = tf.constant([[0.0]], dtype=tf.float32)        # Shape (1, 1)
        # lval_tensor = tf.constant([[self.L_val]], dtype=tf.float32) # Shape (1, 1)
        # x = np.concatenate([zero_tensor, x, lval_tensor], axis=0)
        # return tf.cast(x, dtype=tf.float32)

        interior_points = batch_size - 2 * n_boundary_points
        if interior_points < 0:
            raise ValueError("Batch size is too small for the given number of boundary points. "
                            "Make sure that batch_size >= 2 * n_boundary_points.")
        x_interior = np.random.uniform(0, self.L_val, size=(interior_points, 1))
        x_boundary = np.concatenate([np.zeros((n_boundary_points, 1)), np.ones((n_boundary_points, 1)) * self.L_val], axis=0)
        return tf.cast(x_interior, dtype=tf.float32), tf.cast(x_boundary, dtype=tf.float32)
    
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
    def calculate_loss(self, model:tf.keras.Model, x_interior, x_boundary, aggregate_boundaries:bool=False, training:bool=False):
        W_interior, _, dW_dxx, _, dW_dxxxx = self.derivatives(model, x_interior, training=training)
        f_loss_interior = tf.reduce_mean(((self.q_0) * tf.math.sin(np.pi*x_interior/self.L_val) - self.EI_val*dW_dxxxx)**2)
        W_boundary, _, dW_dxx_boundary, _, dW_dxxxx_boundary = self.derivatives(model, x_boundary, training=training)
        b1_loss = tf.reduce_mean((W_boundary[0:3])**2)  # At x=0
        b2_loss = tf.reduce_mean((W_boundary[-4:-1])**2)  # At x=L
        b3_loss = tf.reduce_mean((dW_dxx_boundary[0:3])**2)  # At x=0
        b4_loss = tf.reduce_mean((dW_dxx_boundary[-4:-1])**2)  # At x=L
        total_loss = f_loss_interior + b1_loss + b2_loss + b3_loss + b4_loss
        return total_loss, f_loss_interior, [b1_loss, b2_loss, b3_loss, b4_loss]
    # def calculate_loss(self, model:tf.keras.Model, x, aggregate_boundaries:bool=False, training:bool=False):
        # W, _, dW_dxx, _, dW_dxxxx = self.derivatives(model, x, training=training)
        # f_loss = tf.reduce_mean(((self.q_0) * tf.math.sin(np.pi*x/self.L_val) - self.EI_val*dW_dxxxx)**2)
        # xl = tf.cast(x < self.TOL, dtype=tf.float32)
        # xu = tf.cast(x > self.L_val - self.TOL, dtype=tf.float32)
        # b1_loss = tf.reduce_mean((xl * W)**2)
        # b2_loss = tf.reduce_mean((xu * W)**2)
        # b3_loss = tf.reduce_mean((xl * dW_dxx)**2)
        # b4_loss = tf.reduce_mean((xu * dW_dxx)**2)
        # return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss]


       
    
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
    
    @tf.function
    def manual(self, model, x_interior, x_boundary, args:dict):
        loss, f_loss, b_losses = self.calculate_loss(model, x_interior, x_boundary, aggregate_boundaries=False, training=True)
        # loss = args['lam'+str(0)] * f_loss + tf.reduce_sum([args['lam'+str(i+1)] * b_losses[i] for i in range(len(b_losses))])
        grads = [tf.gradients(loss, model[0].trainable_variables)]
        args = args.copy()
        for i, loss_val in enumerate([f_loss] + b_losses):
            args['l' + str(i)] = loss_val
        return grads, f_loss, b_losses, args
    # def manual(self, model, x, args:dict):
    #     f_loss, b_losses = self.calculate_loss(model, x, aggregate_boundaries=False, training=True)
    #     loss = args['lam'+str(0)]*f_loss + tf.reduce_sum([args['lam'+str(i+1)]*b_losses[i] for i in range(len(b_losses))])
    #     grads = [tf.gradients(loss, model[0].trainable_variables)]
    #     args = args.copy()
    #     for i, loss in enumerate([f_loss] + b_losses):
    #         args['l'+str(i)] = loss    
    #     return grads, f_loss, b_losses, args

    
    def train(self, nlayers=5, nnodes=360, lr=0.001, epochs=5001, batch_size=1024, resample=True, T=0.1, alpha=0.999, rho=1, patience=4, factor=0.1, capture=1, strategy=True):
        model = [self.fully_connected(nlayers, nnodes)]
        # print(model[0].layers[1].get_weights()[0][:5])  # First 5 weights of the first layer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        args = {"lam"+str(i): tf.constant(1.) for i in range(self.num_b_losses+1)} 
        # args.update({"l"+str(i): tf.constant(1.) for i in range(self.num_b_losses+1)})
        # args.update({"l0"+str(i): tf.constant(1.) for i in range(self.num_b_losses+1)})
        # args["T"] = tf.constant(T, dtype=tf.float32)
        # # rho_schedule = (np.random.uniform(size=epochs+1) < rho).astype(int).astype(np.float32)
        # rho_schedule = (self.rng.uniform(size=epochs+1) < rho).astype(int).astype(np.float32)
        # args['rho'] = tf.constant(rho_schedule[0], dtype=tf.float32)
        # alpha_schedule = [tf.constant(1., tf.float32), tf.constant(0., tf.float32)] + [tf.constant(alpha, tf.float32)]
        # args["alpha"] = alpha_schedule[0]
        alpha = [1.]
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})

        best_loss = 1e9
        cooldown = patience
        plateau_count = 0
        plateau_threshold = 5   # Number of consecutive epochs with minimal improvement to consider a plateau
        min_delta = 1e-13        # Minimum improvement to consider as progress

        print('Start training of PINN using Tensorflow')
        start = time()
        # x = self.training_batch(batch_size)
        x_interior, x_boundary = self.training_batch(batch_size)
        lambdas = []
        losses = []
        previous_loss = None
        

        for epoch in range(epochs):
            if resample:
                x = self.training_batch(batch_size)
            # grads, f_loss, b_losses, args = self.manual(model, tf.constant(x, dtype=tf.float32), args)
            grads, f_loss, b_losses, args = self.manual(model, x_interior, x_boundary, args)
            optimizer.apply_gradients(zip(grads[0], model[0].trainable_variables))

            # self.current_losses = [args['l'+str(i)].numpy() for i in range(self.num_b_losses+1)]
            # if (epoch == 1):
            #     for i in range(self.num_b_losses+1):
            #         args['l0'+str(i)] = ([f_loss]+b_losses)[i]
            # if len(alpha_schedule) > 1:
            #     args['alpha'] = alpha_schedule[1]
            #     alpha_schedule = alpha_schedule[1:]
            # args['rho'] = rho_schedule[1]
            # rho_schedule = rho_schedule[1:]


            if epoch % capture == 0:
                x_val, w_val = self.validation_batch()
                val_loss = self.validation_loss(model, x_val, w_val)
                loss = f_loss + tf.reduce_sum(b_losses)
                x_boundary_vals = tf.constant([[0.0], [self.L_val/2], [self.L_val]], dtype=tf.float32)
                W_boundary_vals = model[0](x_boundary_vals, training=False)
                deflection_at_0 = W_boundary_vals[0].numpy()
                deflection_at_L_mid = W_boundary_vals[1].numpy()
                deflection_at_L = W_boundary_vals[2].numpy()
                print(
                    f"Epoch {epoch:4d} | "
                    f"F: {f_loss.numpy():.1e} | "
                    f"B: {'  '.join(f'{b.numpy():.1e}' for b in b_losses)} | "
                    f"Val Loss: {val_loss.numpy():.1e} | "
                    f"W_0: {deflection_at_0.item():.1e} | "
                    f"W_mid: {deflection_at_L_mid.item():.1e} | "
                    f"W_L: {deflection_at_L.item():.1e} | "
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
