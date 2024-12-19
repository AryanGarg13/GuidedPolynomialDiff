import h5py
import numpy as np
import torch
import time
import os
import sys
import einops
import scipy

sys.path.append(os.getcwd())

class MPiNetPrior:

    def __init__(self, mode = "test") -> None:
        
        # if mode == "train":
        #     self._data = h5py.File("assets/dataset/mpinet/hybrid_solutions/train.hdf5", 'r')
        # elif mode == "test":
        #     self._data = h5py.File("assets/dataset/mpinet/val.hdf5", 'r')

        # self._data_global = self._data['global_solutions']
        # self._data_hybrid = self._data['hybrid_solutions']

        # self.num_traj_gl = self._data_global.shape[0]
        # self.num_traj_hyb = self._data_hybrid.shape[0]
        
        self.n = 50
        self.c = 7


    def sample_trajectories(self, batch_size):

        '''Sample 50% of the samples from global solutions (approx. 3M) and 50% from hybrid solutions (approx. 3M)
        '''
        sample_set1 = self._data_global[np.sort(np.random.choice(self.num_traj_gl, batch_size//2, replace=False)), :, :]
        np.random.shuffle(sample_set1)
        sample_set2 = self._data_hybrid[np.sort(np.random.choice(self.num_traj_hyb, batch_size//2, replace=False)), :, :]
        np.random.shuffle(sample_set2)

        batch = einops.rearrange(np.concatenate((sample_set1, sample_set2), axis=0), 'b n c -> b c n')
        np.random.shuffle(batch)

        return batch
    
    def bernstein_transform(self, num_coeffs):

        # First axis is x from 0 to 1, total 50 points
        # Second axis is the bernstein equations, for v = 0 to 7

        x = np.arange(self.n)[:, np.newaxis]
        x = np.tile(x, (1, num_coeffs)) / (self.n - 1)

        v = np.arange(num_coeffs)[np.newaxis, :]
        v = np.tile(v, (self.n, 1))

        nCv = scipy.special.comb((num_coeffs-1), v)
        BT = nCv * (x**v) * ((1 - x) ** (num_coeffs - 1 - v))
        BT = BT[np.newaxis, :, :]
        self.BT = BT
        # BT = torch.tensor(BT, dtype = torch.float32, device = self.device)

        # # We are flipping this for some reason (but how did it work earlier?):
        # BT = np.swapaxes(BT, -1, -2)
        # self.BT = BT

        return BT
    
    def polynomial_to_trajectory(self, coeffs):

        return np.swapaxes(np.matmul(self.BT, np.swapaxes(coeffs, -1, -2)), -1, -2)
    
    def polynomialMSE(self, coeffs, x_ref, num_coeffs):

        coeffs = np.reshape(coeffs, (x_ref.shape[0], x_ref.shape[1], num_coeffs))
        x_poly = np.matmul(self.BT, np.swapaxes(coeffs, -1, -2))
        x_poly = np.swapaxes(x_poly, -1, -2)

        cost = np.sum(np.mean(np.square(x_poly - x_ref), axis = (-2, -1)))

        return cost
    
    def fit_to_polynomial(self, x_ref, num_coeffs):

        initial_coeffs = np.random.randn(x_ref.shape[0] * x_ref.shape[1] * num_coeffs)

        res = scipy.optimize.minimize(lambda x: self.polynomialMSE(x, x_ref, num_coeffs),
                                        initial_coeffs,
                                        method = 'SLSQP')
        
        x_fit = np.reshape(res.x, (x_ref.shape[0], x_ref.shape[1], num_coeffs))

        return x_fit
    
    def animate_bernstein_denoising(self, env, intermediate_coeffs, wait_time = 0.1):

        T = intermediate_coeffs.shape[0]

        for t in range(T-1, -1, -1):

            coeffs = intermediate_coeffs[t]
            trajectory = self.polynomial_to_trajectory(coeffs[np.newaxis, :])[0]

            env.remove_all_points()
            env.spawn_points(coeffs, "red")
            env.spawn_points(trajectory)

            time.sleep(wait_time)
            




