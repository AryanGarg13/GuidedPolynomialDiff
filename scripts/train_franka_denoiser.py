import os
import sys
import torch
import torch.nn as nn
import wandb
import numpy as np

sys.path.append(os.getcwd())

from lib.models import TemporalUNet, Diffusion
from lib.priors import MPiNetPrior as Prior

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"``

# -------------- Edit these parameters -------------- #

traj_len = 50
T = 256 # Number of diffusion time steps
epochs = 50000
batch_size = 512
time_dim = 32
checkpoint_per = 1000

model_name = "FrankaBernsteinDenoiser"
mode = "train"

enable_wandb = False

# --------------------------------------------------- #

model_path = "assets/" + model_name

if enable_wandb:
    wandb.init(
        project = model_name
    )

trajectory_path="/home/aadi_iiith/Desktop/RRC/IROS24/unguided_trajs.npy"
trajectories= np.load(trajectory_path)
traj_tens = torch.from_numpy(trajectories[0]).to(device=device, dtype=torch.float32)

num_coeffs = 8
x_ref = trajectories[0]
mpinet = Prior(mode)
bernstein_transform = mpinet.bernstein_transform(num_coeffs)
fitted_coeffs = mpinet.fit_to_polynomial(x_ref, num_coeffs)
print(traj_tens.shape)
mse = mpinet.polynomialMSE(fitted_coeffs, x_ref, num_coeffs)
print("Mean Squared Error for ",num_coeffs," coefficients is:", mse)

fintraj = mpinet.polynomial_to_trajectory(fitted_coeffs)
diffusion = Diffusion(T,traj_len=8,num_channels=7, device = device)

fit_coeff = torch.from_numpy(fitted_coeffs).to(device=device, dtype=torch.float32)
print(fit_coeff.shape)
denoiser = TemporalUNet(model_path = model_path, 
                        input_dim = 7, 
                        time_dim = time_dim,
                        device = device, 
                        dims = (32, 64, 128, 256, 512, 512))

optimizer = torch.optim.Adam(denoiser.parameters(), lr = 0.0001)
loss_fn = nn.MSELoss()



# Ensure the model is also on the same precision and device
denoiser = denoiser.to(device=device, dtype=torch.float32)

# print(x0.shape)
for e in range(epochs):
            
    denoiser.train(True)

    # x0 = prior.sample_trajectories(batch_size)

    X, Y_true, t, mean, var = diffusion.generate_q_sample(fit_coeff, condition = True)

    Y_pred = denoiser(X, t)

    optimizer.zero_grad()

    loss = loss_fn(Y_pred, Y_true)

    loss.backward()

    optimizer.step()

    denoiser.losses = np.append(denoiser.losses, loss.item())

    if enable_wandb:
        wandb.log({"Loss": loss.item()})

    print("\rEpoch number = " + str(denoiser.losses.size) + ", Latest epoch loss = " + str(loss.item()), end="", flush=True)

    denoiser.save()
    if (denoiser.losses.size) % checkpoint_per == 0:
        denoiser.save_checkpoint(denoiser.losses.size)
    