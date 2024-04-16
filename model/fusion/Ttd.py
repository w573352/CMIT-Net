
import numpy as np
import math
from torch import nn, einsum
import torch
from einops import rearrange
from .layers import Attention, ACT2FN, PostionEmbedding, get_config, Mlp,Global_Pool

class MultiAttention(nn.Module):
    def __init__(self, config, is_position=False):
        super().__init__()
        self.config = config
        self.is_position = is_position
        self.v_attention1 = Attention(config)
        self.v_attention2 = Attention(config)
        self.v_attention3 = Attention(config)
        if is_position:
            self.pos_embedding_1 = PostionEmbedding(config, types=1)
            self.pos_embedding_2 = PostionEmbedding(config, types=1)
            self.pos_embedding_3 = PostionEmbedding(config, types=1)

    def forward(self, x):
        batch_size, hidden_size, D, W, H = x.shape

        x_1 = rearrange(x, "b c d w h -> (b d) (w h) c")#MHAxy
        x_2 = rearrange(x, "b c d w h -> (b h) (d w) c")#MHAxy
        x_3 = rearrange(x, "b c d w h -> (b w) (d h) c")  # MHAxy
        if self.is_position:
            x_1 = self.pos_embedding_1(x_1)
            x_2 = self.pos_embedding_2(x_2)
            x_3 = self.pos_embedding_3(x_3)


        x_1 = self.v_attention1(x_1)#MHAxy
        x_2 = self.v_attention2(x_2)
        x_3 = self.v_attention3(x_3)



        x_1 = rearrange(x_1, "(b d) (w h) c -> b (d w h) c", d=D, w=W, h=H)
        x_2 = rearrange(x_2, "(b h) (d w) c -> b (d w h) c", d=D, w=W, h=H)
        x_3 = rearrange(x_3, "(b w) (d h) c -> b (d w h) c", d=D, w=W, h=H)


        return x_1 + x_2 + x_3


class TtdBlock(nn.Module):
    def __init__(self, config, is_position=False):
        super(TtdBlock, self).__init__()
        self.config = config
        self.input_shape = config.img_size
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = MultiAttention(config, is_position=is_position)
    def forward(self, x):#[1,128,8,8,8]
        batch_size, hidden_size, D, W, H = x.shape
        #[2, 128, 8, 8, 8] - >[2, 512, 128]
        x = rearrange(x, "b c d w h -> b (d w h) c")
        h = x
        #[2, 512, 128]->[2, 512, 128]
        x = self.attention_norm(x)#层归一化
        #[2, 512, 128]->[2,128,8,8,8]
        x = rearrange(x, "b (d w h) c -> b c d w h", d=D, w=W, h=H)

        x = self.attn(x)#self-attention #[1,128,8,8,8] -> [1,512,128]
        x = x + h#残差连接

        h = x
        #[1,512,128]->[1,512,128]
        x = self.ffn_norm(x)#归一化
        # [1,512,128]->[1,512,128]
        x = self.ffn(x)#MLP
        x = x + h#残差连接
        #[1,512,128]->[2, 128, 512]
        x = x.transpose(-1, -2)
        out_size = (self.input_shape[0] // self.config.patch_size[0],
                    self.input_shape[1] // self.config.patch_size[1],
                    self.input_shape[2] // self.config.patch_size[2],)
        #[2, 128, 512]->[2, 128, 8, 8, 8]
        x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2])).contiguous()
        #[2, 128, 8, 8, 8]->[2, 128, 8, 8, 8]
        return x#[1,128,8,8,8]

class ModelTtd(nn.Module):
    """
        in_channels:  128
        hidden_size:  128
        img_size:  [8, 8, 8]
        num_heads:  8
        mlp_size:  256
        num_layers:  2
        window_size:  (4, 4, 4)
        out_hidden:  False
    """
    def __init__(self, in_channels,
                 hidden_size,
                 img_size,
                 is_position,
                 mlp_size,
                 window_size,
                 num_heads=8,
                 out_hidden=False
                 ):
        super().__init__()
        self.config = get_config(in_channels=in_channels, hidden_size=hidden_size,
                                 patch_size=(1, 1, 1), img_size=img_size, mlp_dim=mlp_size, num_heads=num_heads, window_size=window_size)
        self.block = TtdBlock(self.config, is_position=is_position)

        self.out_hidden = out_hidden

        self.conv = nn.Conv3d(in_channels=in_channels,  # 128
                             out_channels=in_channels,  # 128
                             kernel_size=(1, 1, 1),  # (1, 1, 1)
                             stride=(1, 1, 1))
    def forward(self, x):#x.shape[1, 128, 8, 8, 8]
        """MLP层和归一化要求以token的形式输入"""
        x =self.conv(x)
        x = self.block(x)

        return x#[1,128,8,8,8]



