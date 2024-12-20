import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
import einops

#################################################################################################################
#---------------------------------------------- LOWER LEVEL BLOCKS ---------------------------------------------#
#################################################################################################################

#--------------------------------------------- 1D CONVOLUTION BLOCK --------------------------------------------#

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
        # Why is padding half of kernel size? This makes sure that in the first convolution, half the kernel has zero elements and the other
        # half has the first few elements of the input. The same applies to the last convolution.

    def forward(self, x):

        return self.block(x)

#################################################################################################################
#------------------------------------------ INTERMEDIATE LEVEL BLOCKS ------------------------------------------#
#################################################################################################################

#---------------------------------------------- LINEAR ATTENTION -----------------------------------------------#

class LinearAttention(nn.Module):
    
    def __init__(self, dim, heads=4, dim_head=32):
        
        super().__init__()
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)

        q = q * self.scale
        k = k.softmax(dim = -1)

        # Produces the context of each element in k w.r.t all other elements in v. (weighted sum of every pair of rows possible)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Weighted sum of every pair of columns possible:
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        
        # Recombine the the 3 chunks that where seperated to get back the same dimension as input
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        
        # Convolve back from hidden channels to the original number of channels (dim)
        out = self.to_out(out)

        return out

#------------------------------------------ RESIDUAL CONVOLUTION BLOCK ------------------------------------------#

class ResidualConvolutionBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        # The convolution that forms the residual connection between input and output
        # If the input and the output have the same number of channels, this is an identity matrix.
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x time_embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)

        return out
    
#------------------------------------------- RESIDUAL ATTENTION BLOCK -------------------------------------------#

class ResidualAttentionBlock(nn.Module):

    def __init__(self, dim, eps = 1e-5):
        
        super().__init__()

        self.attention = LinearAttention(dim)

        # Layer Norm Parameters:
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):

        # Layer Norm
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        out = (x - mean) / (var + self.eps).sqrt() * self.g + self.b

        # Attention Layer
        out = self.attention(out)

        # Residual Connection
        out = out + x

        return out

#################################################################################################################
#---------------------------------------------- HIGH LEVEL BLOCKS ----------------------------------------------#
#################################################################################################################

class DownSampler(nn.Module):
    
    def __init__(self, dim_in, dim_out, is_last = False):

        super().__init__()
        
        self.down = nn.ModuleList([ResidualConvolutionBlock(dim_in, dim_out),
                                   ResidualConvolutionBlock(dim_out, dim_out),
                                   nn.Identity(),  # Replace this with ResidualAttentionBlock
                                   nn.Conv1d(dim_out, dim_out, kernel_size = 3, stride = 2, padding = 1) if not is_last else nn.Identity()])

    def forward(self, x):
        
        x = self.down[0](x)
        x = self.down[1](x)
        x = self.down[2](x)   
        out = self.down[3](x)

        return out

class MiddleBlock(nn.Module):
    
    def __init__(self, mid_dim):

        super().__init__()

        self.middle = nn.ModuleList([ResidualConvolutionBlock(mid_dim, mid_dim),
                                     nn.Identity(),  # Replace this with ResidualAttentionBlock
                                     ResidualConvolutionBlock(mid_dim, mid_dim)])

    def forward(self, x):

        x = self.middle[0](x)
        x = self.middle[1](x)
        out = self.middle[2](x)

        return out

class UpSampler(nn.Module):
    
    def __init__(self, dim_in, dim_out, is_last = False):

        super().__init__()

        self.up = nn.ModuleList([ResidualConvolutionBlock(dim_out, dim_in),
                                 ResidualConvolutionBlock(dim_in, dim_in),
                                 nn.Identity(),  # Replace this with ResidualAttentionBlock
                                 nn.ConvTranspose1d(dim_in, dim_in, kernel_size = 4, stride = 2, padding = 1) if not is_last else nn.Identity()])

    def forward(self, x):

        x = self.up[0](x)
        x = self.up[1](x)
        x = self.up[2](x)
        out = self.up[3](x)

        return out