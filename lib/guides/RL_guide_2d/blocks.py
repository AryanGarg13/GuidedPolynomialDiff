import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
import einops

#################################################################################################################
#---------------------------------------------- LOWER LEVEL BLOCKS ---------------------------------------------#
#################################################################################################################

#--------------------------------------------- 1D CONVOLUTION BLOCK --------------------------------------------#

class Conv2dBlock(nn.Module):
    '''
        Conv2d --> LayerNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels h w -> batch channels 1 h w'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 h w -> batch channels h w'),
            nn.Mish(),
        )

    def forward(self, x):

        return self.block(x)
    
#----------------------------------------- SINUSOIDAL POSITION EMBEDDING ---------------------------------------#

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim, device):
        
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, x):

        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb

#------------------------------------------------- TEMPORAL MLP ------------------------------------------------#

class TimeMLP2D(nn.Module):

    def __init__(self, time_embed_dim, out_channels):

        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, out_channels),
            Rearrange('batch t -> batch t 1 1'),      # This is probably extending by a dimension
        )

    def forward(self, t):

        return self.time_mlp(t)
    
class TimeMLP(nn.Module):

    def __init__(self, time_embed_dim, dim_out):

        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, dim_out),
        )

    def forward(self, t):

        return self.time_mlp(t)

#-------------------------------------------- INITIAL TIME EMBEDDING -------------------------------------------#

class TimeEmbedding(nn.Module):

    def __init__(self, dim, device):

        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, device),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):

        return self.time_mlp(t)

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

class ResidualTemporalConvolutionBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, time_embed_dim, kernel_size=5):

        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(inp_channels, out_channels, kernel_size),
            Conv2dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = TimeMLP2D(time_embed_dim, out_channels)

        # The convolution that forms the residual connection between input and output
        # If the input and the output have the same number of channels, this is an identity matrix.
        self.residual_conv = nn.Conv2d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x time_embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)

        return out
    
class ResidualConvolutionBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size=5):

        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(inp_channels, out_channels, kernel_size),
            Conv2dBlock(out_channels, out_channels, kernel_size),
        ])

        # The convolution that forms the residual connection between input and output
        # If the input and the output have the same number of channels, this is an identity matrix.
        self.residual_conv = nn.Conv2d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x inp_channels x horizon ]
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
    
    def __init__(self, dim_in, dim_out):

        super().__init__()
        
        self.down = nn.ModuleList([ResidualConvolutionBlock(dim_in, dim_out),
                                   ResidualConvolutionBlock(dim_out, dim_out),
                                   nn.Conv2d(dim_out, dim_out, kernel_size = 3, stride = 2, padding = 1),
                                   ResidualConvolutionBlock(dim_out, dim_out),
                                   ResidualConvolutionBlock(dim_out, dim_out)])

    def forward(self, x):
        
        x = self.down[0](x)
        x = self.down[1](x)
        x = self.down[2](x)
        x = self.down[3](x)
        out = self.down[4](x)

        return out
    
class DownSamplerTemporal(nn.Module):
    
    def __init__(self, dim_in, dim_out, time_dim, is_last = False):

        super().__init__()
        
        self.down = nn.ModuleList([ResidualTemporalConvolutionBlock(dim_in, dim_out, time_embed_dim = time_dim),
                                   ResidualTemporalConvolutionBlock(dim_out, dim_out, time_embed_dim = time_dim),
                                   ResidualTemporalConvolutionBlock(dim_out, dim_out, time_embed_dim = time_dim),
                                   ResidualTemporalConvolutionBlock(dim_out, dim_out, time_embed_dim = time_dim),
                                   nn.Conv2d(dim_out, dim_out, kernel_size = 3, stride = 2, padding = 1) if not is_last else nn.Identity()])

    def forward(self, x, t):
        
        x = self.down[0](x, t)
        x = self.down[1](x, t)
        x = self.down[2](x, t)
        x = self.down[3](x, t)
        out = self.down[4](x)

        return out

class TemporalMLP(nn.Module):
    
    def __init__(self, dim_in, dim_out, time_dim, hidden_dims = (64, 8)):

        super().__init__()

        self.mlp = nn.ModuleList([])

        self.mlp.append(LinearTemporalLayer(dim_in, hidden_dims[0], time_dim))
        for i in range(len(hidden_dims) - 1):
            self.mlp.append(LinearTemporalLayer(hidden_dims[i], hidden_dims[i+1], time_dim))
        self.mlp.append(nn.Linear(hidden_dims[-1], dim_out))

    def forward(self, x, t):

        for i in range(len(self.mlp) - 1):
            x = self.mlp[i](x, t)

        out = self.mlp[-1](x)

        return out
    
class MLP(nn.Module):
    
    def __init__(self, dim_in, dim_out, hidden_dims = (64, 8)):

        super().__init__()

        self.mlp = nn.ModuleList([])

        self.mlp.append(LinearLayer(dim_in, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.mlp.append(LinearLayer(hidden_dims[i], hidden_dims[i+1]))
        self.mlp.append(nn.Linear(hidden_dims[-1], dim_out))

    def forward(self, x):

        for i in range(len(self.mlp) - 1):
            x = self.mlp[i](x)

        out = self.mlp[-1](x)

        return out
    
class LinearTemporalLayer(nn.Module):

    def __init__(self, dim_in, dim_out, time_dim):

        super().__init__()

        self.time_mlp = TimeMLP(time_dim, dim_in)
        
        self.layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.Tanh()
        ])

    def forward(self, x, t):

        x = x + self.time_mlp(t)
        x = self.layers[0](x)
        x = self.layers[1](x)
        out = self.layers[2](x)

        return out
    
class LinearLayer(nn.Module):

    def __init__(self, dim_in, dim_out):

        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.Tanh()
        ])

    def forward(self, x):

        x = self.layers[0](x)
        x = self.layers[1](x)
        out = self.layers[2](x)

        return out