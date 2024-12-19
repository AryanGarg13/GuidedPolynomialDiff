import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from .gaussian import Gaussian
import time

class Diffusion(Gaussian):
    
    def __init__(self, T, num_channels, traj_len, device, variance_thresh = 0.02):

        self.T = T
        self.c = num_channels
        self.n = traj_len

        self.device = device

        self.beta = self.schedule_variance(variance_thresh)
        self.multivariate_normal = MultivariateNormal(torch.zeros((self.c * self.n), dtype = torch.float32, device = device),
                                                      torch.eye((self.c * self.n), dtype = torch.float32, device = device))

        self.alpha = 1 - self.beta
        self.alpha_bar = self.tensor(list(map(lambda t:np.prod(self.alpha[1:t+1].numpy(force=True)), np.arange(self.T+1))))

        Gaussian.__init__(self)
    
    def tensor(self, x):

        return torch.tensor(x, dtype = torch.float32, device = self.device)

    def cosine_func(self, t):
        """
        Implements the cosine function at a timestep t
        Inputs:
        t -> Integer ; current timestep
        T -> Integer ; total number of timesteps
        Outputs:
        out -> Float ; cosine function output
        """
        
        s = 1e-10
        out = torch.cos((t/self.T + s) * (np.pi/2) / (1 + s)) ** 0.15
        
        return out

    def schedule_variance(self, thresh = 0.02):
        """
        Schedules the variance for the diffuser
        Inputs:
        T      -> Integer ; total number of timesteps
        thresh -> Float ; variance threshold at the last step
        Outputs:
        schedule -> Numpy array of shape (2,) ; the variance schedule
        """
            
        schedule = torch.linspace(0, thresh, self.T + 1, dtype = torch.float32, device = self.device)

        return schedule

    def q_sample(self, x, t, eps = None):
        """
        Generates q(xt+1/xt)
        Inputs:
        x -> Tensor of shape (num_samples, num_channels, trajectory_length)
        t -> Tensor of length (num_samples, ); timesteps (these are the output timesteps)
        eps -> Tensor of shape (num_samples, num_channels, trajectory_length)
        Outputs:
        xt -> Tensor of shape (num_samples, num_channels, trajectory_length)
        mean -> Tensor of shape (num_samples, num_channels, trajectory_length); mean from which sample is taken
        var-> Tensor of length (num_samples); variance from which sample is taken
        """

        # Gather trajectory params:
        b = x.size()[0]
        
        if type(eps) == type(None):
            eps = self.multivariate_normal.sample((b,)).view(b, self.c, self.n)
                    
        mean = torch.sqrt(self.alpha[t].unsqueeze(-1).unsqueeze(-1)) * x
        var = torch.sqrt(1 - self.alpha[t].unsqueeze(-1).unsqueeze(-1))
        xt = mean + var * eps
        
        return xt, mean, var
    
    def q_sample_from_x0(self, x0, t, eps = None):
        """
        Generates q(xt/x0)
        Inputs:
        x0 -> Tensor of shape (num_samples, num_channels, trajectory_length)
        t -> Tensor of length (num_samples, ); timesteps (these are the output timesteps)
        Outputs:
        xt -> Tensor of shape (num_samples, num_channels, trajectory_length)
        mean -> Tensor of shape (num_samples, num_channels, trajectory_length); mean from which sample is taken
        var-> Tensor of length (num_samples); variance from which sample is taken
        """

        # Gather trajectory params:
        b = x0.size()[0]
        
        if type(eps) == type(None):
            eps = self.multivariate_normal.sample((b,)).view(b, self.c, self.n)
        
        mean = torch.sqrt(self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)) * x0
        var = torch.sqrt(1 - self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1))

        xt = mean + var * eps

        return xt, mean, var
    
    def p_sample(self, xt, t, eps):
        """
        Generates reverse probability p(xt-1/xt, eps) with no dependence on x0.
        """

        alpha = self.alpha[t].unsqueeze(-1).unsqueeze(-1)
        
        xt_prev = (xt - torch.sqrt(1 - alpha) * eps) / torch.sqrt(alpha)

        return xt_prev
    
    def predict_eps(self, model, X_t, t):

        return model(X_t, t)
    
    def predict_x0(self, xt, t, eps):
        """If differentiable, make sure xt and eps are tensors, and t is an integer. """

        alpha_bar = self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)

        x0 = (xt - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

        return x0
    
    def p_sample_using_posterior(self, xt, t, eps, sigma_wt = 0.5):
        """
        Generates reverse probability p(xt-1/xt, x0) using posterior mean and posterior variance.
        """        
        
        # Gather trajectory params:
        b = xt.size()[0]

        z = self.multivariate_normal.sample((b,)).view(b, self.c, self.n)

        alpha = self.alpha[t].unsqueeze(-1).unsqueeze(-1)
        alpha_bar = self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        alpha_bar_prev = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1)
        if t > 1:
            sigma = sigma_wt * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - (alpha_bar / alpha_bar_prev))
        else:
            sigma = 0

        x0 = self.predict_x0(xt, t, eps)

        xt_prev = (xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha) + sigma*z   # --> This one works

        return xt_prev, x0

    def forward_diffuse(self, x0, condition = True):
        """
        Forward diffuses the trajectory till the last timestep T
        Inputs:
        x -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        Outputs:
        diffusion_trajs -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); diffused trajectories
        eps -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); Noise added at each timestep
        kl_divs -> Numpy array of shape (Timesteps, num_samples); KL Divergence at each timestep
        """
                
        # Gather trajectory params:
        b = x0.size()[0]
        
        # Initialize the diffusion trajectories across time steps:
        diffusion_trajs = torch.zeros((self.T + 1, b, self.c, self.n), dtype = torch.float32, device = self.device)
        diffusion_trajs[0] = x0

        # Calculate epsilons to forward diffuse current trajectory:
        eps = self.multivariate_normal.sample((b*self.T,)).view(self.T, b, self.c, self.n)
        kl_divs = torch.zeros((self.T, b), dtype = torch.float32, device = self.device)

        # ??? This loop can probably be vectorized
        
        for t in range(1, self.T + 1):

            diffusion_trajs[t] = self.q_sample(diffusion_trajs[t-1], t, eps[t-1])
            if condition:
                diffusion_trajs[t, :, :, 0] = x0[:, :, 0]
                diffusion_trajs[t, :, :, -1] = x0[:, :, -1]
            
            # ???
            kl_divs[t-1, :] = self.KL_divergence_against_gaussian(diffusion_trajs[t].flatten())

        return diffusion_trajs, eps, kl_divs

    def reverse_diffuse(self, xT, eps):
        """
        Reverse diffuses the trajectory from last timestep T to first timestep 0
        Inputs:
        xT  -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        eps -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); Noise added at each timestep
        Outputs:
        diffusion_trajs -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); reverse diffused trajectories
        """
        
        # Gather trajectory params:
        b = xT.size()[0]
        
        diffusion_trajs = torch.zeros((self.T + 1, b, self.c, self.n), dtype = torch.float32, device = self.device)
        diffusion_trajs[self.T] = xT

        for t in range(self.T, 0, -1):

            diffusion_trajs[t-1] = self.p_sample(diffusion_trajs[t], t, eps[t-1])

        return diffusion_trajs

    def generate_q_sample(self, x0, time_steps = None, condition = True):
        """
        Generates q samples for a random set of timesteps, useful for training\n\n
        Inputs:\n
        x0          -> Numpy array of shape (num_samples, num_channels, trajectory_length)\n
        condition   -> Bool ; whether to apply conditioning or not\n
        return_type -> String (either "tensor" or "numpy") ; whether to return a tensor or numpy array\n\n
        Outputs:\n
        X -> Numpy array of shape (num_samples, num_channels, trajectory_length); xt from each x0 diffused to a random timestep t\n
        Y -> Numpy array of shape (num_samples, num_channels, trajectory_length); the noise added to each x0\n
        time_step -> Numpy array of shape (num_samples, ); the timestep to which each x0 is diffused\n
        means -> Numpy array of shape (num_samples, num_channels, trajectory_length); mean of each x0 diffused to a random timestep t\n
        vars -> Numpy array of shape (num_samples, ); the variance at the timestep to which x0 is diffused\n
        """

        # Refer to the Training Algorithm in Ho et al 2020 for the psuedo code of this function
            
        # Gather trajectory params:
        b = x0.size()[0]
        
        if type(time_steps) == type(None):
            time_steps = torch.randint(high = self.T, size = (b,))

        # Remember, for each channel, we get a multivariate normal distribution.
        eps = self.multivariate_normal.sample((b,)).view(b, self.c, self.n)
        
        # Size chart:
        # x0         => (num_samples, 2, traj_len)
        # xt         => (num_samples, 2, traj_len)
        # time_steps => (num_samples,)
        # eps        => (num_samples, 2, traj_len)
        
        xt, means, vars = self.q_sample_from_x0(x0, time_steps, eps)  

        # CONDITIONING:
        if condition:
            xt[:, :, 0] = x0[:, :, 0]
            xt[:, :, -1] = x0[:, :, -1]

        return xt, eps, time_steps, means, vars
    
    def denoise(self, model, num_channels = None, state_dim = None, batch_size = 1, start = None, goal = None, condition = True, return_intermediates = False):
        
        if num_channels == None:
            num_channels = self.c
        if state_dim == None:
            state_dim = self.n

        X_t = self.multivariate_normal.sample((batch_size,)).view(batch_size, num_channels, state_dim)

        if condition:
            X_t[:, :, 0] = start
            X_t[:, :, -1] = goal

        model.train(False)

        if return_intermediates:
            intermediates = torch.zeros((batch_size, self.T + 1, num_channels, state_dim), dtype = torch.float32, device = self.device)
            intermediates[:, self.T, :, :] = X_t

        x0_predictions = torch.zeros((batch_size, self.T + 1, num_channels, state_dim), dtype = torch.float32, device = self.device)
        timestep = torch.arange(self.T + 1, dtype = torch.float32, device = self.device).unsqueeze(-1)

        for t in range(self.T, 0, -1):

            epsilon = model(X_t, timestep[t])

            X_t, X_0 = self.p_sample_using_posterior(X_t, t, epsilon)

            x0_predictions[:, t, :, :] = X_0

            if condition:
                X_t[:, :, 0] = start
                X_t[:, :, -1] = goal

            intermediates[:, t-1, :, :] = X_t

            print("\rDenoised " + str(t-1) + " timesteps", end="", flush=True)

        x0_predictions[:, 0, :, :] = X_t

        if return_intermediates:
            return X_t, x0_predictions, intermediates

        return X_t, x0_predictions
    
    def denoise_guided(self, model, guide, batch_size = 1, guide_weight = 0.1, 
                       start = None, goal = None, condition = True, return_intermediates = False):
        
        X_t = self.multivariate_normal.sample((batch_size,)).view(batch_size, self.c, self.n)

        if condition:
            X_t[:, :, 0] = start
            X_t[:, :, -1] = goal

        model.train(False)

        if return_intermediates:
            intermediates = torch.zeros((batch_size, self.T + 1, self.c, self.n), dtype = torch.float32, device = self.device)
            intermediates[:, self.T, :, :] = X_t

        x0_predictions = torch.zeros((batch_size, self.T + 1, self.c, self.n), dtype = torch.float32, device = self.device)
        timestep = torch.arange(self.T + 1, dtype = torch.float32, device = self.device).unsqueeze(-1)

        for t in range(self.T, 0, -1):

            epsilon = model(X_t, timestep[t])

            # Add guidance here
            if t > 1: # and not stop_guidance:
                
                xt_tensor = X_t[:, :, 1:-1]
                xt_tensor.requires_grad = True
                xt_tensor.grad = None

                x0_tensor = self.predict_x0(xt_tensor, t, epsilon[:, :, 1:-1])
                cost = guide.cost(x0_tensor, start, goal)

                cost.backward()               

                gradient = F.normalize(xt_tensor.grad, dim = -1)

                X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - guide_weight * gradient

            X_t, X_0 = self.p_sample_using_posterior(X_t, t, epsilon)
            x0_predictions[:, t, :, :] = X_0

            if condition:
                X_t[:, :, 0] = start
                X_t[:, :, -1] = goal

            intermediates[:, t-1, :, :] = X_t

            print("\rDenoised " + str(t-1) + " timesteps, " + "cost = " + str(cost), end="", flush=True)

        if return_intermediates:
            return X_t, x0_predictions, intermediates

        return X_t, x0_predictions
    
    def denoise_bernstein_guided(self, model, guide, bernstein_transform, batch_size = 1, 
                                 guide_weight = 0.1, start = None, goal = None, condition = True, 
                                 return_intermediates = False):
        
        num_coeffs = bernstein_transform.shape[-1]
        traj_len = bernstein_transform.shape[-2]
        
        coeffs_t = self.multivariate_normal.sample((batch_size,)).view(batch_size, self.c, num_coeffs)
        BT = torch.tensor(bernstein_transform, dtype = torch.float32, device = self.device)
        
        X_t = torch.transpose(torch.matmul(BT, torch.transpose(coeffs_t, -1, -2)), -1, -2)

        if condition:
            coeffs_t[:, :, 0] = start
            coeffs_t[:, :, -1] = goal

        model.train(False)

        if return_intermediates:
            coeff_inters = torch.zeros((batch_size, self.T + 1, self.c, num_coeffs), dtype = torch.float32, device = self.device)
            coeff_inters[:, self.T, :, :] = coeffs_t
            intermediates = torch.zeros((batch_size, self.T + 1, self.c, traj_len), dtype = torch.float32, device = self.device)
            intermediates[:, self.T, :, :] = X_t

        coeffs_0_predictions = torch.zeros((batch_size, self.T + 1, self.c, num_coeffs), dtype = torch.float32, device = self.device)

        timestep = torch.arange(self.T + 1, dtype = torch.float32, device = self.device).unsqueeze(-1)

        # Time Measurements:
        cost_time = 0
        grad_time = 0
        
        for t in range(self.T, 0, -1):

            epsilon = model(coeffs_t, timestep[t])

            # Add guidance here
            if t > 1: # and not stop_guidance:

                cost_start = time.time()

                coeffs_t.requires_grad = True
                coeffs_t.grad = None

                coeffs_0_tensor = self.predict_x0(coeffs_t, t, epsilon)
                coeffs_0_tensor[:, :, 0] = start
                coeffs_0_tensor[:, :, -1] = goal
                x0_tensor = torch.transpose(torch.matmul(BT, torch.transpose(coeffs_0_tensor, -1, -2)), -1, -2)

                cost = guide.cost(x0_tensor, start, goal)

                cost_time += np.round((time.time() - cost_start) * 1000)

                grad_start = time.time()

                cost.backward()

                grad_time += np.round((time.time() - grad_start) * 1000)

                gradient = F.normalize(coeffs_t.grad, dim = -1)

                with torch.no_grad():
                    coeffs_t = coeffs_t - guide_weight * gradient
                
            with torch.no_grad():
                coeffs_t, coeffs_0 = self.p_sample_using_posterior(coeffs_t, t, epsilon)
                coeffs_0_predictions[:, t, :, :] = coeffs_0

                X_t = torch.transpose(torch.matmul(BT, torch.transpose(coeffs_t, -1, -2)), -1, -2)

            if condition:
                coeffs_t[:, :, 0] = start
                coeffs_t[:, :, -1] = goal

            with torch.no_grad():
                coeff_inters[:, t-1, :, :] = coeffs_t
                intermediates[:, t-1, :, :] = X_t

            print("\rDenoised " + str(t-1) + " timesteps, " + "cost = " + str(cost), end="", flush=True)

        print("It took cost " + str(cost_time) + " milliseconds")
        print("It took grad " + str(grad_time) + " milliseconds")
        
        if return_intermediates:
            return X_t, coeffs_t, coeffs_0_predictions, coeff_inters, intermediates

        return X_t, coeffs_t, coeffs_0_predictions
    
    def phi(self, X_t, eps_t, t, delta = 1):

        alpha_bar = self.alpha_bar[t, np.newaxis, np.newaxis]
        alpha_bar_prev = self.alpha_bar[t-delta, np.newaxis, np.newaxis]

        X_t_coeff = np.sqrt(alpha_bar_prev) / np.sqrt(alpha_bar)
        eps_t_coeff_numerator = alpha_bar_prev - alpha_bar
        eps_t_coeff_denominator = np.sqrt(alpha_bar) * (np.sqrt((1 - alpha_bar_prev) * alpha_bar) + np.sqrt((1 - alpha_bar) * alpha_bar_prev))
        eps_t_coeff = eps_t_coeff_numerator / eps_t_coeff_denominator
        
        return (X_t_coeff * X_t) - (eps_t_coeff * eps_t)
    
    def PRK(self, model, X_t, t, delta = 2):

        eps_t_1 = self.predict_eps(model, X_t, t)
        X_t_1 = self.phi(X_t, eps_t_1, t, delta = delta//2)

        eps_t_2 = self.predict_eps(model, X_t_1, t-delta//2)
        X_t_2 = self.phi(X_t, eps_t_2, t, delta = delta//2)

        eps_t_3 = self.predict_eps(model, X_t_2, t-delta//2)
        X_t_3 = self.phi(X_t, eps_t_3, t, delta = delta)

        eps_t_4 = self.predict_eps(model, X_t_3, t-delta)

        eps_t_dash = (1/6) * (eps_t_1 + 2 * eps_t_2 + 2 * eps_t_3 + eps_t_4)

        X_t = self.phi(X_t, eps_t_dash, t, delta = delta)
        
        return X_t, eps_t_1

    def PLMS(self, model, X_t, t, eps_t_1, eps_t_2, eps_t_3, delta = 1):

        eps_t = self.predict_eps(model, X_t, t)
        eps_t_dash = (1/24) * (55 * eps_t - 59 * eps_t_1 + 37 * eps_t_2 - 9 * eps_t_3)
        X_t = self.phi(X_t, eps_t_dash, t, delta = delta)

        return X_t, eps_t

    def pndm_denoise_guided(self, model, guide, env, traj_len, num_channels, batch_size = 1, guide_weight = 0.1, 
                       early_stop_threshold = 0.01, start = None, goal = None, condition = True, return_intermediates = False):
        
        X_t = np.random.multivariate_normal(mean = np.zeros(traj_len), cov = np.eye(traj_len), size = (batch_size, num_channels))

        if condition:
            X_t[:, :, 0] = start[:]
            X_t[:, :, -1] = goal[:]

        start_tensor = torch.tensor(start, dtype = torch.float32, device = self.device)
        goal_tensor = torch.tensor(goal, dtype = torch.float32, device = self.device)
        # start => (7,)
        # goal => (7,)

        model.train(False)

        if return_intermediates:
            intermediates = np.zeros((batch_size, self.T + 1, num_channels, traj_len))
            intermediates[:, self.T, :, :] = X_t.copy()

        stop_guidance = False

        for t in range(self.T, self.T-3, -1):

            if t == 255:    
                X_t, eps_t_3 = self.PRK(model, X_t, t)
            if t == 254:    
                X_t, eps_t_2 = self.PRK(model, X_t, t)
            if t == 253:    
                X_t, eps_t_1 = self.PRK(model, X_t, t)

            env.remove_all_points()
            env.remove_all_text()
            # env.spawn_points(X_t[0])
            point_color = "dark_green" if stop_guidance else "black"
            env.spawn_points(X_t[0], point_color)
            env.add_timestep_text(t-1, position = [-1, -1, 1], early_stop = stop_guidance)

            # Add guidance here
            xt_tensor = torch.tensor(X_t[:, :, 1:-1], dtype = torch.float32, device = self.device)
            xt_tensor.requires_grad = True
            xt_tensor.grad = None

            cost = guide.cost(xt_tensor, start_tensor, goal_tensor)
            cost.backward()                        

            gradient = xt_tensor.grad.cpu().numpy()
            if np.linalg.norm(gradient) > 0:    
                gradient = gradient / np.linalg.norm(gradient)

            X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - guide_weight * gradient

            if condition:
                X_t[:, :, 0] = start[:]
                X_t[:, :, -1] = goal[:]

            intermediates[:, t-1, :, :] = X_t.copy()

            print("\rDenoised " + str(t-1) + " timesteps, " + "cost = " + str(cost), end="", flush=True)

        for t in range(self.T-3, 0, -1):

            X_t, eps_t = self.PLMS(model, X_t, t, eps_t_1, eps_t_2, eps_t_3)

            z = np.random.multivariate_normal(mean = np.zeros(traj_len), cov = np.eye(traj_len), size = (batch_size, num_channels))

            alpha_bar = self.alpha_bar[t, np.newaxis, np.newaxis]
            alpha_bar_prev = self.alpha_bar[t-1, np.newaxis, np.newaxis]
            if t > 1:
                sigma = 0.3 * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * np.sqrt(1 - (alpha_bar / alpha_bar_prev))
            else:
                sigma = 0
            
            X_t = X_t + sigma * z

            eps_t_3 = eps_t_2
            eps_t_2 = eps_t_1
            eps_t_1 = eps_t

            env.remove_all_points()
            env.remove_all_text()
            # env.spawn_points(X_t[0])
            point_color = "dark_green" if stop_guidance else "black"
            env.spawn_points(X_t[0], point_color)
            env.add_timestep_text(t-1, position = [-1, -1, 1], early_stop = stop_guidance)

            # Add guidance here
            xt_tensor = torch.tensor(X_t[:, :, 1:-1], dtype = torch.float32, device = self.device)
            xt_tensor.requires_grad = True
            xt_tensor.grad = None

            cost = guide.cost(xt_tensor, start_tensor, goal_tensor)
            cost.backward()                        

            gradient = xt_tensor.grad.cpu().numpy()
            if np.linalg.norm(gradient) > 0:    
                gradient = gradient / np.linalg.norm(gradient)

            X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - guide_weight * gradient

            if condition:
                X_t[:, :, 0] = start[:]
                X_t[:, :, -1] = goal[:]

            intermediates[:, t-1, :, :] = X_t.copy()

            print("\rDenoised " + str(t-1) + " timesteps, " + "cost = " + str(cost), end="", flush=True)


        if return_intermediates:
            return X_t, intermediates   

        return X_t

