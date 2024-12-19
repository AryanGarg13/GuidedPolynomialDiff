import os
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class LinearVAE(nn.Module):

    def __init__(self, model_path, num_channels, state_dim, latent_dim, device, dims = (128, 64, 32, 16)):

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

    def __init__(self, num_channels, state_dim, latent_dim, dims, device):

        super().__init__()

        self.encoder = nn.ModuleList([])
        self.encoder.append(Layer(num_channels * state_dim, dims[0]))
        for i in range(len(dims) - 1):
            self.encoder.append(Layer(dims[i], dims[i+1]))

        self.predict_mean = nn.Linear(dims[-1], latent_dim)
        self.predict_log_variance = nn.Linear(dims[-1], latent_dim)    # Intuitively, the log of a variance can be linear, so its better to predict the log instead of variance directly     
        
    def forward(self, x):

        x = x.view(x.size()[0], x.size()[1] * x.size()[2])
        
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        mean = self.predict_mean(x)
        log_variance = self.predict_log_variance(x)
        covariance = torch.diag_embed(torch.exp(log_variance))

        return mean, covariance

class Decoder(nn.Module):

    def __init__(self, num_channels, state_dim, latent_dim, dims, device):
        
        super().__init__()

        self.device = device
        self.num_channels = num_channels
        self.state_dim = state_dim

        self.decoder = nn.ModuleList([])
        self.decoder.append(Layer(latent_dim, dims[-1]))
        for i in range(len(dims) - 1, 0, -1):
            self.decoder.append(Layer(dims[i], dims[i-1]))

        self.decoder.append(Layer(dims[0], num_channels * state_dim))

        self.multivariate_normal = MultivariateNormal(torch.zeros(latent_dim, device = self.device), torch.eye(latent_dim, device = self.device))

    def forward(self, mean = None, covariance = None):

        epsilon = self.multivariate_normal.sample((mean.size()[0],))

        if mean != None and covariance != None:
            z = mean + torch.squeeze(torch.matmul(covariance, epsilon.unsqueeze(-1)), dim = -1)
        else:
            z = epsilon

        x = self.decoder[0](z)
        for i in range(1, len(self.decoder)):
            x = self.decoder[i](x)

        x = x.view(x.size()[0], self.num_channels, self.state_dim)

        return x

class Layer(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            # nn.Dropout(p = 0.2),
            nn.Tanh()
        )

    def forward(self, t):

        return self.layer(t)
    