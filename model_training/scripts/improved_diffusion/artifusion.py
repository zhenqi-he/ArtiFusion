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
from .model import *

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
                # print(layer)
                # print(isinstance(layer, TimestepBlock))
                x = layer(x)
        return x

class SwinTransformerBlock(TimestepBlock):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,time_embed_dim=128*4,out_channels = 64):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = False
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.out_channels = out_channels
        # print("swin out channel ",out_channels)
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                time_embed_dim,
                self.dim,
            ),
        )
        
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            _,mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self,x,emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )
    

    def _forward(self, x,emb):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x.clone()
        # print("xo .shape ",x.shape)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # print("emb shape ",emb.shape)
        # print(self.emb_layers)
        nW, x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(B, -1, self.window_size * self.window_size,C)
        # print('in swinTransformer block, emb before: ',emb.shape)
        time_embed=self.emb_layers(emb).unsqueeze(1)
        time_embed=time_embed.repeat(1,nW,1,1)
        # print('in swinTransformer block, emb after: ',time_embed.shape)
        x_windows = torch.concat((x_windows,time_embed),dim=-2)
        x_windows = x_windows.flatten(0,1)
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # print("x1.shape ",x.shape)
        
        
        # print("emb shape ",emb.shape)
        # emb_out = self.emb_layers(emb).type(x.dtype)
        # while len(emb_out.shape) < len(x.shape):
        #     emb_out = emb_out[..., None]
        # print("emb shape ",emb_out.shape)
        # print("x2.shape ",x.shape)
        # x = x + emb_out
        h = x.view(B, H * W, C)
        
        # FFN
        
        # print("h.shape ",h.shape)
        # print("shortcut.shape ",shortcut.shape)
        drop_path = self.drop_path(h)
        # print("drop_path.shape ",drop_path.shape)
        x = shortcut + drop_path
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class BasicLayer(TimestepBlock):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,time_embed_dim=128*4,out_channels = 64):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = False

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,time_embed_dim=time_embed_dim,out_channels = out_channels)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = TimestepEmbedSequential(downsample(input_resolution, dim=dim, norm_layer=norm_layer))
        else:
            self.downsample = None

    def forward(self, x, emb):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x,emb)
            else:
                x = blk(x,emb)
        if self.downsample is not None:
            x = self.downsample(x,emb)
        return x

class BasicLayer_up(TimestepBlock):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,out_channels = 64):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
           SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,out_channels = out_channels)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = TimestepEmbedSequential(PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer))
        else:
            self.upsample = None

    def forward(self, x,emb):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x,emb)
            else:
                x = blk(x,emb)
        if self.upsample is not None:
            x = self.upsample(x,emb)
        return x
    
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
        final_upsample="expand_first",       
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.channel_mult = [2, 2, 2, 2]
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
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_drop_rate=attn_drop_rate
        self.drop_path_rate=drop_path_rate
        self.norm_layer=norm_layer
        self.ape=ape
        self.patch_norm=patch_norm
        self.use_checkpoint=use_checkpoint
        self.final_upsample=final_upsample
        self.norm_layer=nn.LayerNorm 
        
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
                
            ]
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=self.in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        num_res_blocks = 1
        ### channel_mult == depth
        # print("channel_mult,",channel_mult)
        for i_layer, mult in enumerate(self.depths):
            for _ in range(num_res_blocks):
                layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
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
                               use_checkpoint=use_checkpoint,
                               out_channels = int(64/(pow(2,i_layer))))
                # print("layer basicLayer, ", isinstance(layers,TimestepBlock))
                
                self.input_blocks.append(TimestepEmbedSequential(layers))
                

    
    
        self.output_blocks_layers_up = nn.ModuleList([])
        self.output_blocks_concat_back_dim = nn.ModuleList([])
        # concat_dim = [8192,8192,8192,8192]
        for i_layer, mult in enumerate(self.depths):
            for _ in range(num_res_blocks):
                concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
                # print("{} layer, {}x{}".format(i_layer,2*int(embed_dim*2**(self.num_layers-1-i_layer)),int(embed_dim*2**(self.num_layers-1-i_layer))))
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
                                    use_checkpoint=use_checkpoint,
                                    out_channels = int(64/(pow(2,3-i_layer))))
                self.output_blocks_layers_up.append(TimestepEmbedSequential(layer_up))
                self.output_blocks_concat_back_dim.append(TimestepEmbedSequential(concat_linear))
                
        self.norm = self.norm_layer(self.num_features)
        self.norm_up= self.norm_layer(self.embed_dim)

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
       
    
        # print("timesteps shape ",timesteps.shape)
        # print("input x shape ",x.shape)
        x = self.patch_embed(x)
        # print("input x shape after patch_emb",x.shape)
        x = self.pos_drop(x)
        # print("input x shape after pos_drop",x.shape)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # print('emb shape after time_embed in forward ',emb.shape)
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x.type(self.inner_dtype)
        # print("downsample before, h shape ",h.shape)
        for module in self.input_blocks:
            # print("emb,",emb.shape)
            # print("downsample, h shape ",h.shape)
            hs.append(h)
            h = module(h, emb)
            
            
            
            
            
        h = self.norm(h)
        # print("len hs ", len(hs))
        for inx, layer_up in enumerate(self.output_blocks_layers_up):
            # print("updample h shape ",h.shape)
            if inx == 0:
                h = layer_up(h,emb)
            else:
                # print("updample h shape ",h.shape)
                # print("skipconnection h shape ",hs[3-inx].shape)
                cat_in = th.cat([h, hs[3-inx]], dim=-1)
                # print("cat_in shape ",cat_in.shape)
                # print("emb shape ",emb.shape)
                h = self.output_blocks_concat_back_dim[inx](cat_in,emb)
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
        

        # print("h ",h.shape)
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
                cat_in = th.cat([h, hs[3-inx]], -1)
                h = self.concat_back_dim[inx](h,emb)
                h = layer_up(h,emb)
            result["up"].append(h.type(x.dtype))
        return result

class SuperResModel(ArtiFusionModel):

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

