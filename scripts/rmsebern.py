import h5py
import numpy as np
import torch
import time
import os
import sys
import einops
import scipy
import math  
import sklearn.metrics  
import pybullet as p
import pybullet_data
import time
from tqdm import tqdm

# Connect to the physics server (GUI mode)
# p.connect(p.GUI)

# # Load environment (e.g., a plane or URDF model)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Path to data
# planeId = p.loadURDF("plane.urdf")

# # Set gravity and simulation time step
# p.setGravity(0, 0, -9.8)
# p.setTimeStep(1/240)

# # You can now add objects, robots, and simulate


# sys.path.append(os.getcwd())
# data = h5py.File("/home/aadi_iiith/Desktop/RRC/IROS24/IROS24/assets/dataset/mpinet/val.hdf5",'r')

class MPiNetPrior:

    def __init__(self, mode = "test") -> None:
        
        if mode == "train":
            self._data = h5py.File("/home/aadi_iiith/Desktop/RRC/IROS24/IROS24/assets/dataset/mpinet/hybrid_solutions/train.hdf5", 'r')
        elif mode == "test":
            self._data = np.load("/home/aadi_iiith/Desktop/RRC/IROS24/unguided_trajs.npy")

        # self._data_global = self._data['global_solutions']
        # self._data_hybrid = self._data['hybrid_solutions']

        # self.num_traj_gl = self._data_global.shape[0]
        # self.num_traj_hyb = self._data_hybrid.shape[0]
        
        self._data_global = self._data
        self.n = self._data_global.shape[-1]
        self.c = self._data_global.shape[-2]


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
        # print(f"x shape {x.shape}")
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
        x_poly = self.polynomial_to_trajectory(coeffs)
        # print(f"x_poly shape {self.BT.shape}")

        cost = np.sum(np.mean(np.square(x_poly - x_ref), axis = (-2, -1)))
        return cost
    
    def fit_to_polynomial(self, x_ref, num_coeffs):

        initial_coeffs = np.random.randn(x_ref.shape[0] * x_ref.shape[1] * num_coeffs)
        # print(f"x_ref shape {x_ref.shape}")
        res = scipy.optimize.minimize(lambda x: self.polynomialMSE(x, x_ref, num_coeffs),
                                        initial_coeffs,
                                        method = 'SLSQP')
        
        x_fit = np.reshape(res.x, (x_ref.shape[0], x_ref.shape[1], num_coeffs))

        return x_fit
    



    def animate_bernstein_denoising(self, env, intermediate_coeffs,orig,wait_time = 0.1):

        T = intermediate_coeffs.shape[0]

        for t in range(T-1, -1, -1):

            coeffs = intermediate_coeffs[t]
            print(coeffs.shape)
            trajectory = self.polynomial_to_trajectory(coeffs[np.newaxis, :])[0]
            coeffs = np.transpose(coeffs)
            trajectory = np.transpose(trajectory)
            print(trajectory.shape)
            traj = orig[t]
            traj = np.transpose(traj)
            env.remove_all_points()
            # env.spawn_points(coeffs, "red")
            env.spawn_points(trajectory)
            # env.spawn_points(traj, "blue")
            time.sleep(wait_time)
            

mpinet = MPiNetPrior(mode="test")

# batch_size = 100
# trajectories = mpinet.sample_trajectories(batch_size)
trajectory_path="/home/aadi_iiith/Desktop/RRC/IROS24/unguided_trajs.npy"
trajectories= np.load(trajectory_path)
print(trajectories.shape)
x_ref = trajectories[0]
print(x_ref.shape)

num_coeffs = 8
bernstein_transform = mpinet.bernstein_transform(num_coeffs)
fitted_coeffs = mpinet.fit_to_polynomial(x_ref, num_coeffs)
print(fitted_coeffs.shape)
mse = mpinet.polynomialMSE(fitted_coeffs, x_ref, num_coeffs)
print("Mean Squared Error for ",num_coeffs," coefficients is:", mse)

fintraj = mpinet.polynomial_to_trajectory(fitted_coeffs)
print(fintraj[0][0])
# print(fitted_coeffs.shape)
print(x_ref[0][0])
# class PyBulletEnv:

#     def __init__(self):
#         self.points = []

#     def spawn_points(self, points, color="green"):
#         """Spawn points in the PyBullet environment."""
#         for point in points:
            
#             x = point[0]
#             y = point[1]
#             z = point[2]
#             visual_shape_id = p.createVisualShape(
#                 shapeType=p.GEOM_SPHERE,
#                 radius=0.02,
#                 rgbaColor=[0, 1, 0, 1] if color == "green" else [1, 0, 0, 1]
#             )
#             point_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=[x, y, z])
#             self.points.append(point_id)

#     def remove_all_points(self):
#         """Remove all points from the PyBullet environment."""
#         for point_id in self.points:
#             p.removeBody(point_id)
#         self.points = []

# Initialize PyBullet environment
# env = PyBulletEnv()
# mpinet.animate_bernstein_denoising(env,fitted_coeffs,x_ref,0.1)

# Assuming `mpinet`, `fitted_coeffs`, and `fintraj` are already defined as per your provided code
# Animate the trajectory in PyBullet
# mpinet.animate_bernstein_denoising(env, fitted_coeffs)