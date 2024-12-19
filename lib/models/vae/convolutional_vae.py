import os
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class ConvolutionalVAE(nn.Module):

    def __init__(self, model_path, num_channels, state_dim, latent_dim, device, dims = (4, 8, 16, 32)):

        super().__init__()

        self.device = device
        self.latent_dim = latent_dim

        self.q = Encoder(num_channels, state_dim, latent_dim, dims, device)
        self.p = Decoder(num_channels, state_dim, latent_dim, dims, device)

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
        x_dash = self.p(mean, covariance)

        return x_dash, mean, covariance
    
    def loss(self, x, x_dash, mean, covariance):

        MSE_loss = self.mse_loss(x_dash, x)
        sum_squares_means = torch.sum(torch.square(mean), dim = -1)
        covariance_trace = torch.sum(torch.diagonal(covariance, dim1 = -2, dim2 = -1), dim = -1)
        covariance_logdet = torch.log(torch.det(covariance))
        KL_loss = torch.sum(0.5 * (sum_squares_means + covariance_trace - covariance_logdet - self.latent_dim))

        return MSE_loss + KL_loss
    
    def encode(self, x):

        mean, covariance = self.q(x)
        epsilon = self.p.multivariate_normal.sample((mean.size()[0],))

        z = mean + torch.mul(covariance, epsilon)

        return z
    
    def decode(self, num_samples = 1, mean = None, covariance = None):

        if mean == None and covariance == None:
            mean = torch.zeros((num_samples, self.latent_dim), dtype = torch.float32, device = self.device)
            covariance = torch.eye(self.latent_dim, dtype = torch.float32, device = self.device).unsqueeze(0).expand(num_samples, -1, -1)

        x_dash = self.p(mean, covariance)

        return x_dash

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

    def __init__(self, inp_channels, state_dim, latent_dim, dims, device):

        super().__init__()

        self.encoder = nn.ModuleList([])
        in_size = state_dim

        self.encoder.append(DownSamplingLayer(inp_channels, dims[0], in_size))
        in_size = in_size//4
        for i in range(len(dims) - 2):
            self.encoder.append(DownSamplingLayer(dims[i], dims[i+1], in_size))
            in_size = in_size//4

        self.encoder.append(DownConvBlock(dims[-2], dims[-1], out_size = in_size//2))

        self.predict_mean = nn.Linear(dims[-1], latent_dim)
        self.predict_log_variance = nn.Linear(dims[-1], latent_dim)    # Intuitively, the log of a variance can be linear, so its better to predict the log instead of variance directly     
        
    def forward(self, x):

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        x = torch.reshape(x, (x.size()[0], x.size()[1] * x.size()[2]))
        mean = self.predict_mean(x)
        log_variance = self.predict_log_variance(x)
        covariance = torch.diag_embed(torch.exp(log_variance))

        return mean, covariance

class Decoder(nn.Module):

    def __init__(self, inp_channels, state_dim, latent_dim, dims, device):
        
        super().__init__()

        self.device = device
        self.c = inp_channels
        self.n = state_dim

        self.linear = LinearLayer(latent_dim, dims[-1])

        self.decoder = nn.ModuleList([])
        in_size = state_dim // (4**(len(dims) - 1))
        self.final_size = in_size // 2
        self.final_channels = dims[-1]

        self.decoder.append(UpConvBlock(dims[-1], dims[-2], in_size))

        for i in range(len(dims) - 2, 0, -1):
            self.decoder.append(UpSamplingLayer(dims[i], dims[i-1], in_size))
            in_size = in_size * 4

        self.decoder.append(UpSamplingLayer(dims[0], inp_channels, in_size))
        in_size = in_size * 4

        self.final_conv = nn.Conv1d(inp_channels, inp_channels, kernel_size = 5, padding = 2)

        self.multivariate_normal = MultivariateNormal(torch.zeros(latent_dim, device = self.device), torch.eye(latent_dim, device = self.device))

    def forward(self, mean = None, covariance = None):

        epsilon = self.multivariate_normal.sample((mean.size()[0],))

        if mean != None and covariance != None:
            z = mean + torch.squeeze(torch.matmul(covariance, epsilon.unsqueeze(-1)), dim = -1)
        else:
            z = epsilon

        x = self.linear(z)
        x = torch.reshape(x, (x.size()[0], self.final_channels, self.final_size))
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)

        out = self.final_conv(x)

        return out

class DownSamplingLayer(nn.Module):

    def __init__(self, inp_channels, out_channels, in_size):

        super().__init__()
        
        self.block = nn.ModuleList([
            DownConvBlock(inp_channels, out_channels, out_size = in_size//2),
            DownConvBlock(out_channels, out_channels, out_size = in_size//4)
        ])

        self.residual_block = nn.Conv1d(inp_channels, out_channels, kernel_size = 9, stride = 4, padding = 4)

    def forward(self, x):

        out = self.block[0](x)
        out = self.block[1](out) + self.residual_block(x)

        return out
    
class UpSamplingLayer(nn.Module):

    def __init__(self, inp_channels, out_channels, in_size):

        super().__init__()
        
        self.block = nn.ModuleList([
            UpConvBlock(inp_channels, out_channels, out_size = in_size*2),
            UpConvBlock(out_channels, out_channels, out_size = in_size*4)
        ])

        self.residual_block = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size = 6, stride = 4, padding = 1)

    def forward(self, x):

        out = self.block[0](x)
        out = self.block[1](out) + self.residual_block(x)

        return out

class DownConvBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, out_size, kernel_size = 5, stride = 2):

        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding = kernel_size // 2, stride = stride),
            nn.LayerNorm([out_channels, out_size]),
            nn.Mish()
        )

    def forward(self, x):

        return self.block(x)
    
class UpConvBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, out_size, kernel_size = 4, stride = 2, padding = 1):

        super().__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose1d(inp_channels, out_channels, kernel_size, stride = stride, padding = padding),
            nn.LayerNorm([out_channels, out_size]),
            nn.Mish()
        )

    def forward(self, x):

        return self.block(x)
    
class LinearLayer(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )

    def forward(self, t):

        return self.layer(t)
    

    