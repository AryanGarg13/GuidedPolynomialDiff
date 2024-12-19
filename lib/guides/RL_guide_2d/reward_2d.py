import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.transforms.functional as tvtf

from .blocks import *

class Reward2D(nn.Module):

    def __init__(self, model_path, in_channels, state_size, device, conv_dims = (32, 64), mlp_dims = (64, 8)):

        super().__init__()

        self.device = device
        # !!! Start Here - Down Sampler and MLP are defined already

        conv_dims = [in_channels, *conv_dims]  # length of dims is 5

        # Down Sampling:
        self.down_samplers = nn.ModuleList([])
        for i in range(len(conv_dims) - 1):
            self.down_samplers.append(DownSampler(conv_dims[i], conv_dims[i+1]))

        final_size = ((state_size // (2**(len(conv_dims) - 1)))**2) * conv_dims[-1]

        self.mlp = MLP(final_size, 1, mlp_dims)
        self.sigmoid = nn.Sigmoid()
        
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
        """
        x => Tensor of size (batch_size, 2, state_size, state_size)
        """
        
        for i in range(len(self.down_samplers)):
            x = self.down_samplers[i](x)

        x = torch.reshape(x, (x.size()[0], -1))

        x = self.mlp(x)

        out = self.sigmoid(x)

        return out

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

    

            
