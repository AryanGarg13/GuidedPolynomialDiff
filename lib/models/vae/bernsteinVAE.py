import os
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import scipy

from .blocks import *

class BernsteinVAE(nn.Module):

    def __init__(self, model_path, num_channels, state_dim, latent_dim, num_coeffs, device, dims = (32, 64, 128, 256)):

        super().__init__()

        dims = [num_channels, *dims]  # length of dims is 5
        
        self.device = device
        self.state_dim = state_dim
        self.num_coeffs = num_coeffs
        self.latent_dim = latent_dim
        self.final_size = int(np.ceil(state_dim / (2**(len(dims) - 2))))

        self.BT = self.bernstein_transform()

        self.q = Encoder(self.final_size, latent_dim, dims, device)
        self.p = Decoder(latent_dim, num_coeffs, self.BT, dims, device)

        self.mse_loss = nn.MSELoss()

        self.model_path = model_path
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            self.losses = np.array([])
        elif len(os.listdir(model_path)) == 0:
            self.losses = np.array([])
        else:
            self.load()

        _ = self.to(device)

    def forward(self, x):

        mean, covariance = self.q(x)
        x_dash, _ = self.p(mean, covariance)

        return x_dash, mean, covariance
    
    def bernstein_transform(self):

        # First axis is x from 0 to 1, total 50 points
        # Second axis is the bernstein equations, for v = 0 to 7

        x = np.arange(self.state_dim)[:, np.newaxis]
        x = np.tile(x, (1, self.num_coeffs)) / (self.state_dim - 1)

        v = np.arange(self.num_coeffs)[np.newaxis, :]
        v = np.tile(v, (self.state_dim, 1))

        nCv = scipy.special.comb((self.num_coeffs-1), v)
        BT = nCv * (x**v) * ((1 - x) ** (self.num_coeffs - 1 - v))
        BT = BT[np.newaxis, :, :]
        BT = torch.tensor(BT, dtype = torch.float32, device = self.device)

        return BT
    
    def loss(self, x, x_dash, mean, covariance, KL_wt = 0.1):

        MSE_loss = self.mse_loss(x_dash, x)
        sum_squares_means = torch.sum(torch.square(mean), dim = -1)
        covariance_trace = torch.sum(torch.diagonal(covariance, dim1 = -2, dim2 = -1), dim = -1)
        covariance_logdet = torch.log(torch.det(covariance))
        KL_loss = torch.mean(0.5 * (sum_squares_means + covariance_trace - covariance_logdet - self.latent_dim))

        return MSE_loss + KL_wt * KL_loss
    
    def encode(self, x):

        mean, covariance = self.q(x)
        epsilon = self.p.multivariate_normal.sample((mean.size()[0],))

        z = mean + torch.mul(covariance, epsilon)

        return z
    
    def decode(self, num_samples = 1, mean = None, covariance = None):

        if mean == None and covariance == None:
            mean = torch.zeros((num_samples, self.latent_dim), dtype = torch.float32, device = self.device)
            covariance = torch.eye(self.latent_dim, dtype = torch.float32, device = self.device).unsqueeze(0).expand(num_samples, -1, -1)

        x_dash, coeffs = self.p(mean, covariance)

        return x_dash, coeffs

    def freeze(self):

        for param in self.parameters():
            if param.requires_grad:
                param.requires_grad = False
    
    def save(self):

        torch.save(self.state_dict(), self.model_path + "/weights_latest.pt")
        np.save(self.model_path + "/losses.npy", self.losses)

    def save_checkpoint(self, checkpoint):
        
        torch.save(self.state_dict(), self.model_path + "/weights_" + str(checkpoint) + ".pt")
        np.save(self.model_path + "/latest_checkpoint.npy", checkpoint)
    
    def load(self):

        self.losses = np.load(self.model_path + "/losses.npy")
        self.load_state_dict(torch.load(self.model_path + "/weights_latest.pt", map_location = self.device))
        print("Loaded Model at " + str(self.losses.size) + " epochs")

    def load_checkpoint(self, checkpoint):

        _ = input("Press Enter if you are running the model for inference, or Ctrl+C\n(Never load a checkpoint for training! This will overwrite progress)")
        
        latest_checkpoint = np.load(self.model_path + "/latest_checkpoint.npy")
        self.load_state_dict(torch.load(self.model_path + "/weights_" + str(checkpoint) + ".pt"))
        self.losses = np.load(self.model_path + "/losses.npy")[:checkpoint]

class Encoder(nn.Module):

    def __init__(self, final_size, latent_dim, dims, device):

        super().__init__()

        self.encoder = nn.ModuleList([])
        
        # Down Sampling:
        self.encoder = nn.ModuleList([])
        for i in range(len(dims) - 2):      # Loops 0, 1, 2
            self.encoder.append(DownSampler(dims[i], dims[i+1]))
        self.encoder.append(DownSampler(dims[-2], dims[-1], is_last = True))  # 3 -> 4

        self.encoder.append(MiddleBlock(dims[-1]))

        self.predict_mean = nn.Linear(dims[-1] * final_size, latent_dim)
        self.predict_log_variance = nn.Linear(dims[-1] * final_size, latent_dim)    # Intuitively, the log of a variance can be linear, so its better to predict the log instead of variance directly     
        
    def forward(self, x):

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        x = torch.reshape(x, (x.size()[0], x.size()[1] * x.size()[2]))
        mean = self.predict_mean(x)
        log_variance = self.predict_log_variance(x)
        covariance = torch.diag_embed(torch.exp(log_variance))

        return mean, covariance

class Decoder(nn.Module):

    def __init__(self, latent_dim, num_coeffs, BT, dims, device):
        
        super().__init__()

        self.device = device

        self.num_coeffs = num_coeffs
        self.BT = BT
        self.final_channels = dims[-1]

        self.linear = LinearLayer(latent_dim, dims[-1] * num_coeffs)

        self.decoder = nn.ModuleList([])
        self.decoder.append(ResidualConvolutionBlock(dims[-1], dims[-1]))
        self.decoder.append(ResidualConvolutionBlock(dims[-1], dims[-1]))
        for i in range(len(dims) - 1, 1, -1):  # Loops 4, 3, 2  since the last one is a seperate convolution
            self.decoder.append(ResidualConvolutionBlock(dims[i], dims[i-1]))
            self.decoder.append(ResidualConvolutionBlock(dims[i-1], dims[i-1]))

        # Final Convolution:
        self.final_conv = nn.Sequential(Conv1dBlock(dims[1], dims[1], kernel_size = 5),
                                        nn.Conv1d(dims[1], dims[0], kernel_size = 1))

        self.multivariate_normal = MultivariateNormal(torch.zeros(latent_dim, device = self.device), torch.eye(latent_dim, device = self.device))

    def forward(self, mean = None, covariance = None):

        epsilon = self.multivariate_normal.sample((mean.size()[0],))

        if mean != None and covariance != None:
            z = mean + torch.squeeze(torch.matmul(covariance, epsilon.unsqueeze(-1)), dim = -1)
        else:
            z = epsilon

        x = self.linear(z)
        x = torch.reshape(x, (x.size()[0], self.final_channels, self.num_coeffs))
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)

        coeffs = self.final_conv(x)
        # Expected shape of out before transformation is (b, 7, 8)
        # (50, 8) * (8, 7)
        out = torch.matmul(self.BT, torch.transpose(coeffs, -1, -2))
        out = torch.transpose(out, -1, -2)
        # Now we should get (b, 7, 50)

        return out, coeffs

class LinearLayer(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )

    def forward(self, x):

        return self.layer(x)
    

    