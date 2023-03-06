from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)

import copy
import logging
import math

from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import os

import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
import scipy.io as sio
import matplotlib.pyplot as plt
from model import *

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
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
    """
    
class ArtiFusionModel(nn.Module):
    def __init__(
        self,
        in_channels, # in_chans
        model_channels, # = 128 ?
        out_channels, # 3 or 6（learn sigma）
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        
        img_size=256,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        depths_decoder=[1, 2, 2, 2], 
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4., 
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
        
        
    ):
         super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_layers = len(channel_mult)
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        
        self.img_size=img_size
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.depths=depths
        self.depths_decoder=depths_decoder
        self.num_heads=num_heads
        self.window_size=window_size
        self.mlp_ratio=mlp_ratio
        self.qkv_bias=qkv_bias
        self.qk_scale=qk_scale
        self.drop_rate=drop_rate
        self.attn_drop_rate=attn_drop_rate
        self.drop_path_rate=drop_path_rate
        self.norm_layer=norm_layer
        self.ape=ape
        self.patch_norm=patch_norm
        self.use_checkpoint=use_checkpoint
        self.final_upsample=final_upsample
        
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        
            
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
                )
            ]
        )
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        ### channel_mult == depth
        for i_layer, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=self.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                               drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)]
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                

    
    
        self.output_blocks_layers_up = nn.ModuleList([])
        self.output_blocks_concat_back_dim = nn.ModuleList([])
        for i_layer, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer ==0 :
                    layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                        patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                    depth=depths[(self.num_layers-1-i_layer)],
                                    num_heads=num_heads[(self.num_layers-1-i_layer)],
                                    window_size=self.window_size,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                    use_checkpoint=use_checkpoint)
                self.output_blocks_layers_up.append(TimestepEmbedSequential(*layer_up))
                self.output_blocks_concat_back_dim.append(TimestepEmbedSequential(*concat_linear))
                
    self.norm = norm_layer(self.num_features)
    self.norm_up= norm_layer(self.embed_dim)
    
    if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=embed_dim,out_channels=self.out_channels,kernel_size=1,bias=False))
                
    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks_layers_up.apply(convert_module_to_f16)
        self.output_blocks_concat_back_dim.apply(convert_module_to_f16)
        
        self.norm.apply(convert_module_to_f16)
        self.norm_up.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks_layers_up.apply(convert_module_to_f32)
        self.output_blocks_concat_back_dim.apply(convert_module_to_f32)
        
        self.norm.apply(convert_module_to_f32)
        self.norm_up.apply(convert_module_to_f32)
    
    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype
    
    
    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x.type(self.inner_dtype)
        
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.norm(h)
        for inx, layer_up in enumerate(self.output_blocks_layers_up):
            if inx == 0:
                h = layer_up(h,emb)
            else:
                cat_in = th.cat([h, hs.pop()], dim=1)
                h = self.output_blocks_concat_back_dim[inx](h,emb)
                h = layer_up(h,emb)
        
        
        h = h.type(x.dtype)
        h = self.norm_up(h)
        
        H, W = self.patches_resolution
        B, L, C = h.shape
        assert L == H*W, "input features has wrong size"
        
        if self.final_upsample=="expand_first":
            h = self.up(h)
            h = h.view(B,4*H,4*W,-1)
            h = h.permute(0,3,1,2) #B,C,H,W
            h = self.output(h)
        

       
        return h
    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for inx,module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
            if inx != len(self.input_blocks)-1:
                result["down"].append(h.type(x.dtype))
            else:
                result["middle"] = h.type(x.dtype)
            
            
        # h = self.middle_block(h, emb)
        # result["middle"] = h.type(x.dtype)
      
        for inx, layer_up in enumerate(self.output_blocks_layers_up):
            if inx == 0:
                h = layer_up(h,emb)
            else:
                cat_in = th.cat([h, hs.pop()], dim=1)
                h = self.concat_back_dim[inx](h,emb)
                h = layer_up(h,emb)
            result["up"].append(h.type(x.dtype))
        return result

    class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

