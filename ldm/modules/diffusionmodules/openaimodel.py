import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
sys.path.append('....')
from abc import abstractmethod
from functools import partial
import math
from typing import Iterable,List,Union,Optional
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.models.attention import PositionalEncoder
from ldm.modules.attention import FeedForward
from einops import rearrange
import torch
from torch.utils.checkpoint import checkpoint
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer,CrossAttention,default,SpatialVideoTransformer
from ldm.modules.diffusionmodules.util import AlphaBlender
#dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

device = 'cuda:0' if th.cuda.is_available() else 'cpu'

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim:int,
            embed_dim:int,
            num_heads_channels:int,
            output_dim:int=None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim,spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1,embed_dim,3*embed_dim,1)
        self.c_proj = conv_nd(1,embed_dim,output_dim or embed_dim,1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self,x):
        b,c,*_spatial = x.shape
        x = x.reshape(b,c,-1) #NC(HW)
        x = th.cat([x.mean(dim=-1,keepdim=True),x],dim=-1) #NC(HW+1)
        x = x + self.positional_embedding[None,:,:].to(x.dytpe) # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:,:,0]
    
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self,x,emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass

#TODO:whether add temporal attention and gated self attention
class TimestepEmbedSequential(nn.Sequential,TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self,x,emb,boxes_emb=None,text_emb=None):
        for layer in self:
            if isinstance(layer,TimestepBlock):
                x = layer(x,emb)
            elif isinstance(layer,SpatialTransformer):
                if boxes_emb is None:
                    x = layer(x,text_emb=text_emb)
                else:
                    x = layer(x,boxes_emb,text_emb)
            else:
                x = layer(x)
        return x

class TimestepEmbedSequential_Video(nn.Sequential,TimestepBlock):
    def forward(self,
                x:torch.Tensor,
                emb:torch.Tensor,
                context:Optional[torch.Tensor]=None,
                time_context:Optional[int]=None,
                num_frames:Optional[int]=None):
        for layer in self:
            if isinstance(layer,VideoResnetBlock):
                x = layer(x,emb,num_frames)
            elif isinstance(layer,TimestepBlock):
                x = layer(x,emb)
            elif isinstance(layer,SpatialVideoTransformer):
                x = layer(x,context,time_context,num_frames)
            elif isinstance(layer,SpatialTransformer):
                x = layer(x,context)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self,channels,use_conv,dims=2,out_channels=None,padding=1,third_up=False,outpadding=None,scale_factor=2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.outpadding = outpadding
        self.third_up = third_up
        self.scale_factor = scale_factor
        if use_conv:
            self.conv = conv_nd(dims,self.channels,self.out_channels,3,padding=padding)

    def forward(self,x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            t_factor = 1 if not self.third_up else self.scale_factor
            x = F.interpolate(
                x,
                (
                    t_factor * x.shape[2],
                    x.shape[3] * self.scale_factor,
                    x.shape[4] * self.scale_factor
                ),
                mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
            # x = F.interpolate(x,scale_factor=2,mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x
    
class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self,channels,out_channels=None,ks=4,out_padding=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2,padding=1,output_padding=out_padding)
    def forward(self,x):
        return self.up(x)
    
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self,channels,use_conv,dims=2,out_channels=None,padding=1,third_down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims!=3 else ((1,2,2) if not third_down else (2,2,2))
        if use_conv:
            self.op = conv_nd(
                dims,self.channels,self.out_channels,3,stride=stride,padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims,kernel_size=stride,stride=stride)

    def forward(self,x):
        assert x.shape[1] == self.channels
        return self.op(x)

class VResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels: int,
            emb_channels: int,
            dropout: float,
            out_channels: Optional[int] = None,
            use_conv: bool = False,
            use_scale_shift_norm: bool = False,
            dims: int = 2,
            use_checkpoint: bool = False,
            up: bool = False,
            down: bool = False,
            kernel_size: int = 3,
            exchange_temb_dims: bool = False,
            skip_t_emb: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(emb_channels, self.emb_out_channels)
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
            )
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.

        :return: an [N x C x ...] Tensor of outputs.
        """

        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb,use_reentrant=False)
        else:
            return self._forward(x, emb)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...").contiguous()
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            out_padding=None,
            up=False,
            down=False,
            outpadding=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims,channels,self.out_channels,3,padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels,False,outpadding=outpadding)
            self.x_upd = Upsample(channels,False,outpadding=outpadding)
        elif down:
            self.h_upd = Downsample(channels,False,dims)
            self.x_upd = Downsample(channels,False,dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels,2*self.out_channels if use_scale_shift_norm else self.out_channels)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims,self.out_channels,self.out_channels,3,padding=1)
            )
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims,channels,self.out_channels,3,padding=1
            )
        else:
            self.skip_connection = conv_nd(dims,channels,self.out_channels,1)

    def forward(self,x,emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward,x,emb,use_reentrant=False
        )
        
    def _forward(self,x,emb):
        if self.updown:
            in_rest,in_conv = self.in_layers[:-1],self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(x)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[...,None]
        if self.use_scale_shift_norm:
            out_norm,out_rest = self.out_layers[0],self.out_layers[1:]
            scale,shift = th.chunk(emb_out,2,dim=1)
            h = out_norm(h) * (1+scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class ResnetBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels: int,
            emb_channels: int,
            dropout: float,
            out_channels: Optional[int] = None,
            use_conv: bool = False,
            use_scale_shift_norm: bool = False,
            dims: int = 2,
            use_checkpoint: bool = False,
            up: bool = False,
            down: bool = False,
            kernel_size: int = 3,
            exchange_temb_dims: bool = False,
            skip_t_emb: bool = False,
            causal: bool = False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, causal=causal)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(emb_channels, self.emb_out_channels)
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, causal=causal)
            )
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.

        :return: an [N x C x ...] Tensor of outputs.
        """

        if self.use_checkpoint:
            return checkpoint(self._forward,x,emb,use_reentrant=False)
        else:
            return self._forward(x, emb)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...").contiguous()
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert(
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1,channels,channels*3,1)
        if use_new_attention_order:
            #split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            #split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)
        
        self.proj_out = zero_module(conv_nd(1,channels,channels,1))

    def forward(self,x):
        return checkpoint(self._forward,x,use_reentrant=False) # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
    
    def _forward(self,x):
        b,c,*spatial = x.shape
        x = x.reshape(b,c,-1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x+h).reshape(b,c,*spatial)
    

def count_flops_attn(model,_x,y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b,c,*spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2 ) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self,n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self,qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs,width,length = qkv.shape
        assert width % (3*self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q,k,v = qkv.reshape(bs*self.n_heads,ch*3,length).split(ch,dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            'bct,bcs -> bts',q*scale,k*scale
        ) #More stable with f16 than dividing afterwards
        a = th.einsum('bts,bcs->bct',weight,v)
        return a.reshape(bs,-1,length)
    
    @staticmethod
    def count_flops(model,_x,y):
        return count_flops_attn(model,_x,y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self,n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self,qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs,width,length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q,k,v = qkv.chunk(3,dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs*self.n_heads,ch,length),
            (k * scale).view(bs*self.n_heads,ch,length),
        )
        weight = th.softmax(weight.float(),dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct",weight,v.reshape(bs*self.n_heads,ch,length))
        return a.reshape(bs,-1,length)
    
    @staticmethod
    def count_flops(model,_x,y):
        return count_flops_attn(model,_x,y)
    
# class TemporalAttention(nn.Module):
#     def __init__(self,
#                 channels,
#                 num_heads=1,
#                 num_head_channels=-1,
#                 use_checkpoint=False,
#                 use_new_attention_order=False,):
#         super().__init__()
#         self.attention_block = AttentionBlock(channels,num_heads,num_head_channels,use_checkpoint,use_new_attention_order)
    
#     def forward(self,x):
#         bn,c,h,w = x.shape
#         x = rearrange(x,'(b n) c h w -> (b h w) n c')
#         positional_encoder = PositionalEncoder(c,).to(x.device)
#         x = x + positional_encoder(x)
#         x = rearrange(x,'b n c -> b c n')
#         out = self.attention_block(x)
#         out = rearrange(out,'b c n -> b n c')
#         return out
    
# class GatedSelfAttention(nn.Module):
#     def __init__(self, query_dim, context_dim,  n_heads, d_head,use_checkpoint=False,use_new_attention_order=False):
#         super().__init__()
        
#         # we need a linear projection since we need cat visual feature and obj feature
#         self.linear = nn.Linear(context_dim, query_dim)

#         self.attn = AttentionBlock(query_dim, n_heads, d_head,use_checkpoint=use_checkpoint,use_new_attention_order=use_new_attention_order)
#         self.ff = FeedForward(query_dim, glu=True)

#         self.norm1 = nn.LayerNorm(query_dim)
#         self.norm2 = nn.LayerNorm(query_dim)

#         self.register_parameter('alpha_attn', nn.Parameter(th.tensor(0.)) )
#         self.register_parameter('alpha_dense', nn.Parameter(th.tensor(0.)) )

#         # this can be useful: we can externally change magnitude of tanh(alpha)
#         # for example, when it is set to 0, then the entire model is same as original one 
#         self.scale = 1  


#     def forward(self, x, objs):

#         N_visual = x.shape[1]
#         objs = self.linear(objs)
#         h = self.norm1(th.cat([x,objs],dim=1))
#         h = rearrange(h,'b n c -> b c n')
#         h = self.scale * th.tanh(self.alpha_attn) * self.attn(h)
#         h = rearrange(h,'b n c -> b c n')
#         x = x + h[:,0:N_visual,:]
#         x = x + self.scale*th.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
#         return x 

# class UnetBlock(nn.Module):
#     def __init__(self,in_channels,context_dim,num_heads,num_head_channels,use_checkpoint=False,use_new_attention_order=False,
#                  is_train_stage1_step1=False):
#         super().__init__()
#         self.in_channels = in_channels
#         self.context_dim = context_dim
#         self.attn1 = TemporalAttention(in_channels,num_heads=num_heads,num_head_channels=num_head_channels,
#                                        use_checkpoint=use_checkpoint,
#                                        use_new_attention_order=use_new_attention_order)
#         self.attn2 = GatedSelfAttention(query_dim=in_channels,context_dim=context_dim,
#                                         n_heads=num_heads,d_head=num_head_channels,
#                                         use_checkpoint=use_checkpoint,
#                                         use_new_attention_order=use_new_attention_order)
#         self.attn3 = CrossAttention(query_dim=in_channels,context_dim=context_dim,heads=num_heads,
#                                     dim_head=num_head_channels)
#         self.norm1 = nn.LayerNorm(in_channels)
#         self.norm2 = nn.LayerNorm(in_channels)
#         self.norm3 = nn.LayerNorm(in_channels)

#         self.is_train_stage1_step1 = is_train_stage1_step1
#         if is_train_stage1_step1:
#             for params in self.attn1.parameters():
#                 params.requires_grad=False
#     def forward(self,x,boxes_emb,text_emb):
#         b,c,h,w = x.shape
#         x = rearrange(x,'b c h w -> b (h w) c')
#         x = self.attn1(self.norm1(x)) + x
#         x = self.attn2(self.norm2(x),boxes_emb) + x
#         x = self.attn3(self.norm3(x),text_emb) + x
#         x = rearrange(x,'b (h w) c -> b c h w',h=h,w=w)
#         return x



# class UnetModel(nn.Module):
#     """
#     The full UNet model with attention and timestep embedding.
#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param num_res_blocks: number of residual blocks per downsample.
#     :param attention_resolutions: a collection of downsample rates at which
#         attention will take place. May be a set, list, or tuple.
#         For example, if this contains 4, then at 4x downsampling, attention
#         will be used.
#     :param dropout: the dropout probability.
#     :param channel_mult: channel multiplier for each level of the UNet.
#     :param conv_resample: if True, use learned convolutions for upsampling and
#         downsampling.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param num_classes: if specified (as an int), then this model will be
#         class-conditional with `num_classes` classes.
#     :param use_checkpoint: use gradient checkpointing to reduce memory usage.
#     :param num_heads: the number of attention heads in each attention layer.
#     :param num_heads_channels: if specified, ignore num_heads and instead use
#                                a fixed channel width per attention head.
#     :param num_heads_upsample: works with num_heads to set a different number
#                                of heads for upsampling. Deprecated.
#     :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
#     :param resblock_updown: use residual blocks for up/downsampling.
#     :param use_new_attention_order: use a different attention pattern for potentially
#                                     increased efficiency.
#     """
#     def __init__(
#             self,
#             image_size,
#             in_channels,
#             model_channels,
#             out_channels,
#             num_res_blocks,
#             attention_resolutions,
#             dropout=0,
#             channel_mult=(1,2,4,8),
#             conv_resample=True,
#             dims=2,
#             padding_list=[(1,0),(1,0)],
#             num_classes=None,
#             use_checkpoint=False,
#             use_fp16=False,
#             num_heads=-1,
#             num_head_channels=-1,
#             num_heads_upsample=-1,
#             use_scale_shift_norm=False,
#             resblock_updown=False,
#             use_new_attention_order=False,
#             use_spatial_transformer=False,
#             transformer_depth=1,
#             context_dim=None,
#             n_embed=None,
#             legacy=True,
#     ):
#         super().__init__()
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)
            
#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads
        
#         if num_heads == -1:
#             assert num_head_channels != -1,'Either num_heads or num_head_channels has to be set'

#         if num_head_channels == -1:
#             assert num_heads != -1,"Either num_heads or num_head_channels has to be set"

#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_res_blocks = num_res_blocks
#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.num_classes = num_classes
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#         self.predict_codebook_ids = n_embed is not None
#         time_embed_dim = model_channels * 4
#         self.padding_list = padding_list
#         self.time_embed = nn.Sequential(
#             linear(model_channels,time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim,time_embed_dim)
#         )

#         if self.num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes,time_embed_dim)
        
#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims,in_channels,model_channels,3,padding=1)
#                 )
#             ]
#         )
#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level,mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims = dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     layers.append(
#                         UnetBlock(ch,context_dim,num_heads,dim_head)
#                     )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch

#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,
#                                  dims=dims,use_checkpoint=use_checkpoint,
#                                  use_scale_shift_norm=use_scale_shift_norm,
#                                  down=True if level % 2 == 1 else False
#                         )
#                         if resblock_updown
#                         else Downsample(ch,conv_resample,dims=dims,out_channels=out_ch)
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else UnetBlock(ch,context_dim,num_heads,dim_head),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm)
#             )
#         self._feature_size += ch
#         self.output_blocks = nn.ModuleList()
#         for level,mult in list(enumerate(channel_mult))[::-1]:
#             for i in range(num_res_blocks+1):
#                 ich = input_block_chans.pop()
#                 layers = [
#                     ResBlock(
#                         ch+ich,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=model_channels*mult,
#                         dims = dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = model_channels * mult
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     layers.append(
#                         AttentionBlock(
#                             ch,use_checkpoint=use_checkpoint,
#                             num_heads=num_heads_upsample,
#                             num_head_channels=dim_head,
#                             use_new_attention_order=use_new_attention_order
#                         ) if not use_spatial_transformer else UnetBlock(ch,context_dim,num_heads,dim_head)
#                     )
#                 if level % 2 == 0 and level and i == num_res_blocks:
#                     out_ch = ch
#                     layers.append(
#                         ResBlock(
#                             ch,time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             out_padding=padding_list.pop(),
#                             up=True if level % 2 == 0 and level else False
#                         )
#                         if resblock_updown
#                         else Upsample(ch,conv_resample,dims=dims,out_channels=out_ch)
#                     )
#                     ds //= 2
#                 self.output_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch
#         self.out = nn.Sequential(
#             normalization(ch),
#             nn.SiLU(),
#             zero_module(conv_nd(dims,model_channels,out_channels,3,padding=1))
#         )
#         if self.predict_codebook_ids:
#             self.id_predictor = nn.Sequential(
#             normalization(ch),
#             conv_nd(dims, model_channels, n_embed, 1),
#             #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
#         )
    
#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)
#         self.output_blocks.apply(convert_module_to_f16)
    
#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)
#         self.output_blocks.apply(convert_module_to_f32)

#     def forward(self,x,timesteps=None,boxes_emb=None,text_emb=None,y=None,**kwargs):
#         assert (y is not None) == (
#             self.num_classes is not None
#         ),"must specify y if and only if the model is class-conditional"
#         hs = []
#         t_emb = timestep_embedding(timesteps,self.model_channels,repeat_only=False)
#         emb = self.time_embed(t_emb)

#         if self.num_classes is not None:
#             assert y.shape == (x.shape[0],)
#             emb = emb + self.label_emb(y)
        
#         h = x.type(self.dtype)
#         for module in self.input_blocks:
#             h = module(h,emb,boxes_emb,text_emb)
#             hs.append(h)

#         h = self.middle_block(h,emb,boxes_emb,text_emb)
#         for module in self.output_blocks:
#             h = th.cat([h,hs.pop()],dim=1)
#             h = module(h,emb,boxes_emb,text_emb)
            
#         h = h.type(x.dtype)
#         if self.predict_codebook_ids:
#             return self.id_predictor(h)
#         else:
#             return self.out(h)

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    def __init__(
        self,embeddiing_size: int = 256,scale:float=1.0,set_W_to_weight=True,log=True,flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2,embeddiing_size) * scale,requires_grad=False)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            self.W = nn.Parameter(torch.randn(embeddiing_size) * scale,requires_grad=False)
            self.weight = self.W
        
    def forward(self,x):
        if self.log:
            x = torch.log(x)
        x_proj = x * self.weight * 2 * np.pi
        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj),torch.sin(x_proj)],dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj),torch.cos(x_proj)],dim=-1)
        return out

def get_timestep_embedding(timesteps:torch.Tensor,embedding_dim:int,flip_sin_to_cos:bool=False,downscale_freq_shift:float=1,scale:float=1,max_period:int=10000):
     """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
     
     assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
     half_dim = embedding_dim // 2
     exponent = math.log(max_period) * torch.arange(
         start=0,end=half_dim,dtype=torch.float32,device=timesteps.device
     )
     exponent = exponent / (half_dim - downscale_freq_shift)
     emb = torch.exp(exponent)
     emb = timesteps[:,None].float() * emb[None,:]
     emb = scale * emb
     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
     if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
     if embedding_dim % 2 == 1:
         emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
     return emb


class TimeSteps(nn.Module):
    def __init__(self,num_channels:int,flip_sin_to_cos:bool,downscale_freq_shift:float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
    def forward(self,timesteps):
        t_emb = get_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift
        )
        return t_emb

class VideoResnetBlock(ResnetBlock):
    def __init__(
            self,
            channels: int,
            emb_channels: int,
            dropout: float,
            video_kernel_size: Union[int, List[int]] = 3,
            merge_strategy: str = "fixed",
            merge_factor: float = 0.5,
            out_channels: Optional[int] = None,
            use_conv: bool = False,
            use_scale_shift_norm: bool = False,
            dims: int = 2,
            use_checkpoint: bool = False,
            up: bool = False,
            down: bool = False
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down
        )
        self.time_stack = ResnetBlock(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
            causal=False
        )
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern="b t -> b 1 t 1 1"
        )

    def forward(
            self,
            x: torch.Tensor,
            emb: torch.Tensor,
            num_frames: int
    ) -> torch.Tensor:
        x = super().forward(x, emb)

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames).contiguous()
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames).contiguous()

        x = self.time_stack(
            x, rearrange(emb, "(b t) ... -> b t ...", t=num_frames).contiguous()
        )
        x = self.time_mixer(x_spatial=x_mix, x_temporal=x)
        x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        return x

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        movie_len = None,
        height=None,
        width=None,
        ckpt_path=None,
        outpadding=None,
        obj_dims=None,
        ignore_keys=None,
        flip_sin_to_cos=True,
        freq_shift=0,
        class_embed_dim = 4,
        modify_keys=None,
        use_cond_mask=False,
        use_attn_additional=False,
        use_only_action=False,
        action_dim=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.movie_len = movie_len
        self.height = height
        self.width = width
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        if self.num_classes is not None:
            #self.time_proj = TimeSteps(time_embed_dim,flip_sin_to_cos,freq_shift)
            self.label_emb = nn.Sequential(
                linear(class_embed_dim,time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim,time_embed_dim)
            )
            # self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        if use_cond_mask:
            self.cond_time_stack_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if use_only_action:
            self.action_embed = nn.Sequential(
                linear(action_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        self.use_only_action = use_only_action
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,movie_len=self.movie_len,
                            height=self.height,width=self.width,obj_dims=obj_dims,use_attn_additional=use_attn_additional,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                self.height = (self.height+1) // 2
                self.width = (self.width+1) // 2
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,movie_len=self.movie_len,
                            height=self.height,width=self.width,obj_dims=obj_dims,use_attn_additional=use_attn_additional,
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,movie_len=self.movie_len,
                            height=self.height,width=self.width,obj_dims=obj_dims,use_attn_additional=use_attn_additional
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            outpadding=outpadding[level-1]
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch,outpadding=outpadding[level-1])
                    )
                    self.height = self.height * 2 + outpadding[level-1][0]
                    self.width = self.width * 2 + outpadding[level-1][1]
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def from_pretrained_model(self,model_path):
        if not os.path.exists(model_path):
            raise RuntimeError(f'{model_path} does not exist')
        state_dict = torch.load(model_path,map_location='cpu')['state_dict']
        my_model_state_dict = self.state_dict()
        for param in state_dict.keys():
            param = param[22:]
            if param not in my_model_state_dict.keys():
                print("Missing Key:"+str(param))
        #torch.save(my_model_state_dict,'./my_model_state_dict.pt')
        self.load_state_dict(state_dict,strict=False)

    def forward(self, x, timesteps=None, y=None,boxes_emb=None,text_emb=None,cond_mask=None,actions=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if cond_mask is not None and cond_mask.any():
            cond_mask = cond_mask[:,None].float()
            emb = self.cond_time_stack_embed(t_emb) * cond_mask + self.time_embed(t_emb) * (1 - cond_mask)
        else:
            emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            # assert y.shape == (x.shape[0],)
            # if len(y.shape) == 1:
            #     y = y[None,:].expand(x.shape[0],y.shape[0])
            emb = emb + self.label_emb(y)
        if self.use_only_action:
            emb = emb + self.action_embed(actions)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            if boxes_emb is None:
                h = module(h,emb,text_emb=text_emb)
            else:
                h = module(h, emb, boxes_emb,text_emb)
            hs.append(h)
        h = self.middle_block(h, emb, boxes_emb,text_emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            if boxes_emb is None:
                h = module(h,emb,text_emb=text_emb)
            else:
                h = module(h, emb, boxes_emb,text_emb)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
        

class VideoUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult: List[int] = (1,2,4,8),
        conv_resample: bool=True,
        dims: int=2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool=False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int],int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "learned_with_images",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = 'softmax',
        video_kernel_size: Union[int,List[int]] = 3,
        use_linear_in_transformer: bool = False,
        disable_temproal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        add_lora: bool = False,
        action_control: bool = False,
        safetensor_path: str=None,
        ckpt_path: str=None,
        class_embed_dim = 4,
        ignore_keys=None,
        use_multimodal=False,
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth,int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(transformer_depth_middle,transformer_depth[-1])

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels,time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim,time_embed_dim)
        )
        self.cond_time_stack_embed = nn.Sequential(
            linear(model_channels,time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim,time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Sequential(
                linear(class_embed_dim,time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim,time_embed_dim)
            )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential_Video(
                    conv_nd(dims,in_channels,model_channels,3,padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=1,
                context_dim=None,
                use_checkpoint=False,
                disabled_sa=False,
                add_lora=False,
                action_control=False,
                use_multimodal=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in = extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                use_checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temproal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
                add_lora=add_lora,
                action_control=action_control,
                use_multimodal=use_multimodal
            )
        
        def get_resblock(
                merge_factor,
                merge_strategy,
                video_kernel_size,
                ch,
                time_embed_dim,
                dropout,
                out_ch,
                dims,
                use_checkpoint,
                use_scale_shift_norm,
                down=False,
                up=False,
        ):
            return VideoResnetBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up
            )
        
        for level,mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult*model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                            add_lora=add_lora,
                            action_control=action_control,
                            use_multimodal=use_multimodal
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential_Video(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential_Video(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(ch,conv_resample,dims=dims,out_channels=out_ch,third_down=time_downup)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

            if num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // num_head_channels
                dim_head = num_head_channels
        
        self.middle_block = TimestepEmbedSequential_Video(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
                add_lora=add_lora,
                action_control=action_control,
                use_multimodal=use_multimodal
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList(list())
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                            add_lora=add_lora,
                            action_control=action_control,
                            use_multimodal=use_multimodal
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, third_up=time_downup)
                    )

                self.output_blocks.append(TimestepEmbedSequential_Video(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)
            )
        )

        if safetensor_path is not None:
            self.init_from_safetensor(safetensor_path)
    
    def init_from_safetensor(self,safetensor_path):
        from safetensors import safe_open
        sd = dict()
        with safe_open(safetensor_path,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key.startswith('model.diffusion_model'):
                    sd[key[len('model.diffusion_model')+1:]] = f.get_tensor(key)

        missing,unexpected = self.load_state_dict(sd,strict=False)
        print(f"missing:{missing},unexpected:{unexpected}")

    def forward(
            self,
            x:torch.Tensor,
            timesteps:torch.Tensor,
            context: Optional[torch.Tensor]=None,
            y:Optional[torch.Tensor]=None,
            time_context:Optional[torch.Tensor]=None,
            cond_mask:Optional[torch.Tensor] = None,
            num_frames: Optional[int] = None,
    ):
        assert (y is not None) == (
                self.num_classes is not None
        ), "Must specify y if and only if the model is class-conditional"
        hs = list()
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if cond_mask is not None and cond_mask.any():
            cond_mask_ = cond_mask[..., None].float()
            emb = self.cond_time_stack_embed(t_emb) * cond_mask_ + self.time_embed(t_emb) * (1 - cond_mask_)
        else:
            emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                time_context=time_context,
                num_frames=num_frames
            )
            hs.append(h)

        h = self.middle_block(
            h,
            emb,
            context=context,
            time_context=time_context,
            num_frames=num_frames
        )

        for module in self.output_blocks:
            h = torch.cat((h, hs.pop()), dim=1)
            h = module(
                h,
                emb,
                context=context,
                time_context=time_context,
                num_frames=num_frames
            )

        h = h.type(x.dtype)
        return self.out(h)