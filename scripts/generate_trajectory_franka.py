import numpy as np
import torch
import time
import os
import sys

sys.path.append(os.getcwd())

from lib.models import Diffusion, TemporalUNet
from lib.envs import RobotEnvironment
from lib.priors import MPiNetPrior as Prior
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------- User editable parameters -------------- #

T = 255

gui = True
manipulator = True
mpinet_index = 7586 
mode = "train"
model_path = "/home/aadi_iiith/Desktop/RRC/IROS24/IROS24/assets/FrankaBernsteinDenoiser"


# ------------------------------------------------------ #

env = RobotEnvironment(gui, manipulator)

print(env.c)
num_coeffs = 8
diffusion = Diffusion(T,traj_len=num_coeffs,num_channels=7, device = device)

denoiser = TemporalUNet(model_path = model_path,
                        dims=(32, 64, 128, 256, 512, 512),
                        input_dim = env.c,
                        time_dim = 32,
                        device = device)
mpinet = Prior(mode)
bernstein_transform = mpinet.bernstein_transform(num_coeffs)
# obstacle_config, cuboid_config, cylinder_config, start_joints, goal_joints = env.get_mpinet_scene(mpinet_index)
trajectory_path="/home/aadi_iiith/Desktop/RRC/IROS24/unguided_trajs.npy"
trajectories= np.load(trajectory_path)
x0 = torch.from_numpy(trajectories[0][0]).to(device=device, dtype=torch.float32)
start_joints = x0[:,0]
goal_joints = x0[:,-1]
print(start_joints.shape)
# env.spawn_cuboids(cuboid_config, color = "brown")
# env.spawn_cylinders(cylinder_config, color = "brown")

# os.system("clear")
print("Environment and Model Loaded \n")

trajectories, x0_predictions, intermediates = diffusion.denoise(model = denoiser,
                                                                # traj_len = env.n,
                                                                # num_channels = env.c,
                                                                batch_size = 1,
                                                                start = start_joints,
                                                                goal = goal_joints,
                                                                condition = True,
                                                                return_intermediates = True)


trajectory = trajectories[0]
# trajectory = trajectory
print(f" coeff Stuff {trajectory.shape}")
trajectory_numpy = trajectory.detach().cpu().numpy()  # Move to CPU and convert to NumPy
fintraj = mpinet.polynomial_to_trajectory(trajectory_numpy)
fintraj = fintraj.reshape(7,50)
print(f"Final Trajectory Stuff {fintraj.shape}")

env.execute_trajectory(fintraj)
env.execute_trajectory(trajectory_numpy,coeffs=True)

_ = input("Press Enter to Exit")