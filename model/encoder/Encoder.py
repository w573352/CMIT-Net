import torch
import torch.nn as nn
from model.fusion.layers import get_config
import torch.nn.functional as F
from model.fusion.CMT import ModelCMT
from typing import Sequence

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super(Convolution, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride_size, padding_size),
                                    nn.InstanceNorm3d(out_channels),
                                    nn.ReLU())
    def forward(self, x):
        x = self.conv_1(x)
        return x

class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super(TwoConv, self).__init__()
        self.conv_1 = Convolution(in_channels, out_channels, kernel_size, stride_size, padding_size)
        self.conv_2 = Convolution(out_channels, out_channels, kernel_size, stride_size, padding_size)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
            self,
            in_chns: int,#x的通道 解码器
            cat_chns: int,#x_e的通道 编码器
            out_chns: int,#x_e的通道
            pool_size = (2, 2, 2),
            is_last = False
    ):

        super().__init__()
        if is_last == True:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2
        self.upsample = torch.nn.ConvTranspose3d(in_chns, up_chns, kernel_size=pool_size, stride=pool_size, padding=0)

        self.convs = TwoConv(cat_chns + up_chns, out_chns, 3, 1, 1)
        self.channel_mlp1 = nn.Sequential(
            nn.Linear(cat_chns, cat_chns // 2),
            nn.ReLU(),
            nn.Linear(cat_chns // 2, cat_chns),
        )
        self.channel_mlp2 = nn.Sequential(
            nn.Linear(up_chns, up_chns // 2),
            nn.ReLU(),
            nn.Linear(up_chns // 2, up_chns),
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial_conv1_2 = nn.Conv3d(in_channels=cat_chns,
                    out_channels=1,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1))
        self.spatial_conv2_2 = nn.Conv3d(in_channels=up_chns,
                                       out_channels=1,
                                       kernel_size=(1, 1, 1),
                                       stride=(1, 1, 1))
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor, x_e: torch.Tensor):#x.shapetorch.Size([1, 128, 8, 8, 8]) x_e.shapetorch.Size([1, 64, 16, 16, 16])
        #x是解码器的特征 x_e是编码器的特征

        #[1, 128, 8, 8, 8]->[1, 64, 16, 16, 16]
        x_0 = self.upsample(x)
        residual_x_e = x_e
        residual1_x_0 = x_0
        """编码器x_e生成通道注意力权重"""
        #全局平均池化
        #[1, 64, 16, 16, 16]->[1, 64, 1, 1, 1]
        x_e_avg_pool = F.adaptive_avg_pool3d(x_e, (1, 1, 1))

        #[1, 64, 1, 1, 1]->[1,1,1,1,64]
        x_e_avg_pool = x_e_avg_pool.permute(0, 2, 3, 4, 1)
        #MLP
        #[1,1,1,1,64]->[1, 1, 1, 1, 64]
        x_e_avg_pool = self.channel_mlp1(x_e_avg_pool)
        #全局最大池化
        #[1, 64, 16, 16, 16]->[1, 64, 1, 1, 1]
        x_e_max_pool = F.adaptive_max_pool3d(x_e,(1,1,1))
        #[1, 64, 1, 1, 1]->[1,1,1,1,64]
        x_e_max_pool = x_e_max_pool.permute(0, 2, 3, 4, 1)
        #MLP
        #[1,1,1,1,64]->[1, 1, 1, 1, 64]
        x_e_max_pool = self.channel_mlp1(x_e_max_pool)
        #[1,1,1,1,64]
        x_e_c_weight = x_e_max_pool + x_e_avg_pool
        """解码器x_0生成通道注意力权重"""
        # 全局平均池化
        # [1, 64, 16, 16, 16]->[1, 64, 1, 1, 1]
        x_0_avg_pool = F.adaptive_avg_pool3d(x_0, (1, 1, 1))

        # [1, 64, 1, 1, 1]->[1,1,1,1,64]
        x_0_avg_pool = x_0_avg_pool.permute(0, 2, 3, 4, 1)
        # MLP
        # [1,1,1,1,64]->[1, 1, 1, 1, 64]
        x_0_avg_pool = self.channel_mlp2(x_0_avg_pool)
        # 全局最大池化
        # [1, 64, 16, 16, 16]->[1, 64, 1, 1, 1]
        x_0_max_pool = F.adaptive_max_pool3d(x_0, (1, 1, 1))
        # [1, 64, 1, 1, 1]->[1,1,1,1,64]
        x_0_max_pool = x_0_max_pool.permute(0, 2, 3, 4, 1)
        # MLP
        # [1,1,1,1,64]->[1, 1, 1, 1, 64]
        x_0_max_pool = self.channel_mlp2(x_0_max_pool)
        # [1,1,1,1,64]
        x_0_c_weight = x_0_max_pool + x_0_avg_pool
        """融合通道权重"""
        #[1,1,1,1,64] [1,1,1,1,64] ->[1,1,1,1,64]
        c_weight = x_e_c_weight + x_0_c_weight
        c_weight = c_weight.permute(0, 4, 1, 2, 3)
        c_weight = self.sigmoid(c_weight)
        #解码器特征和通道权重相乘
        #[1, 64, 16, 16, 16] * [1,64,1,1,1]->[1, 64, 16, 16, 16]
        x_0 = x_0 * c_weight
        residual2_x_0 = x_0
        """编码器特征x_e生成空间注意力权重"""
        #[1, 64, 16, 16, 16]->[1, 64, 16, 16, 16]
        #x_e_s_weight = self.spatial_conv1_1(x_e)
        #[1, 64, 16, 16, 16]->[1, 1, 16, 16, 16]
        x_e_s_weight = self.spatial_conv1_2(x_e)
        """解码器特征x_0生成空间注意力权重"""
        # [1, 64, 16, 16, 16]->[1, 64, 16, 16, 16]
        #x_0_s_weight = self.spatial_conv2_1(x_0)
        x_0_s_weight = self.spatial_conv2_2(x_0)
        """融合空间权重"""
        s_weight = x_e_s_weight + x_0_s_weight
        s_weight = self.sigmoid(s_weight)
        #[1, 64, 16, 16, 16]*[1, 1, 16, 16, 16]->[1, 64, 16, 16, 16]
        x_0 = x_0 * s_weight
        """residual1_x_0是原始解码器特征 residual2_x_0是经过通道注意力的解码器特征"""
        x_0 = self.relu(residual1_x_0 + residual2_x_0 + x_0)
        #[1, 64, 16, 16, 16] [1, 64, 16, 16, 16]->[1, 128, 16, 16, 16]->[1, 64, 16, 16, 16]
        x = self.convs(torch.cat([residual_x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        in_channels = config.in_channels
        patch_size = config.patch_size

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.norm = LayerNormChannel(num_channels=config.hidden_size)

    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        x = self.norm(x)

        return x



class Conv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 img_size,
                 patch_size,
                 mlp_size=256,
                 num_layers=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = get_config(in_channels=in_channels,
                                 hidden_size=out_channels,
                                 patch_size=patch_size,
                                 mlp_dim=mlp_size,
                                 img_size=img_size)


        self.embeddings = Embeddings(self.config)
        self.gelu = nn.GELU()

    def forward(self, x, out_hidden=False):
        """
        self.embeddings:nn.Conv3d + LayerNormChannel
        """
        x = self.embeddings(x)
        x = self.gelu(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            features: Sequence[int],#fea = (16, 16, 32, 64, 128, 16),
            pool_size,

    ):
        super().__init__()

        fea = features
        self.drop = nn.Dropout()
        self.in_channels = in_channels
        self.features = features
        self.img_size = img_size
        self.model_num = 4
        self.conv_0_list = nn.ModuleList([TwoConv(in_channels, features[0], 3, 1, 1) for i in range(4)])
        #fea[0]输入通道 fea[1]输出通道 patch_size卷积核 步长
        self.down_1_list = nn.ModuleList([Conv(fea[0], fea[1], img_size=img_size[0],patch_size=pool_size[0], mlp_size=fea[1] * 2, num_layers=2)for i in range(4)])
        #[2, 4,16, 64, 64, 64]
        # [2, 4,16, 64, 64, 64]
        self.cmt1 = ModelCMT(model_num=self.model_num,  # 4
                                                                 in_channels=fea[1],  # 16
                                                                 hidden_size=fea[1],  # 16
                                                                 img_size=[64,64,64],  # [64, 64, 64]
                                                                 img_dim=64,  # 64
                                                                 mlp_size=2*fea[1],  # 32
                                                                 )
        self.down_2_list = nn.ModuleList([Conv(fea[1], fea[2], img_size=img_size[1],patch_size=pool_size[1], mlp_size=fea[2] * 2, num_layers=2)for i in range(4)])
        # flair_2.shape [2, 32, 32, 32, 32]
        # #flair_2.shape [2, 32, 32, 32, 32]
        self.cmt2 = ModelCMT(model_num=self.model_num,  # 4
                                              in_channels=fea[2],  # 32
                                              hidden_size=fea[2],  # 32
                                              img_size=[32, 32, 32],  # [32, 32, 32]
                                              img_dim=32,  # 32
                                              mlp_size=2 * fea[2],  # 64
                                              )
        self.down_3_list = nn.ModuleList([Conv(fea[2], fea[3], img_size=img_size[2],patch_size=pool_size[2], mlp_size=fea[3] * 2, num_layers=2)for i in range(4)])
        # flair_3.shape [2, 64, 16, 16, 16]
        self.cmt3 = ModelCMT(model_num=self.model_num,  # 4
                                              in_channels=fea[3],  # 64
                                              hidden_size=fea[3],  # 64
                                              img_size=[16, 16, 16],  # [16, 16, 16]
                                              img_dim=16,  # 16
                                              mlp_size=2 * fea[3],  # 128
                                              )

        self.down_4_list = nn.ModuleList([Conv(fea[3], fea[4], img_size=img_size[3],patch_size=pool_size[3], mlp_size=fea[4] * 2, num_layers=2)for i in range(4)])
        # flair_4.shape [2, 128, 8, 8, 8]
        self.cmt4 = ModelCMT(model_num=self.model_num,#4
                                             in_channels=fea[4],#128
                                             hidden_size=fea[4],#128
                                             img_size=[8,8,8],
                                              img_dim=8,
                                             mlp_size=2*fea[4],#256
                                             )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):##x的维度为[1,4,1,128,128,128]
        # flair.shape [2, 1, 128, 128, 128]
        t1 = x[:, 0]
        t1ce = x[:, 1]
        t2 = x[:, 2]
        flair = x[:, 3]
        """第0次下采样"""
        # flair0 [2, 16, 128, 128, 128]
        #[2, 1, 128, 128, 128]->[2, 16, 128, 128, 128]
        t1_0 = self.conv_0_list[0](t1)
        t1ce_0 = self.conv_0_list[1](t1ce)
        t2_0 = self.conv_0_list[2](t2)
        flair_0 = self.conv_0_list[3](flair)

        """第1次下采样"""
        # flair_1 shape [2, 16, 64, 64, 64]
        #[2, 16, 128, 128, 128]->[1, 16, 64, 64, 64]
        t1_1 = self.down_1_list[0](t1_0)
        t1ce_1 = self.down_1_list[1](t1ce_0)
        t2_1 = self.down_1_list[2](t2_0)
        flair_1 = self.down_1_list[3](flair_0)
        #[2, 16, 64, 64, 64]...->[2,4, 16, 64, 64, 64]
        x = torch.stack([t1_1,t1ce_1,t2_1,flair_1],dim=1)
        """模态间"""
        # 对四个模态进行融合 x:[1, 4, 16, 64, 64, 64]->[1, 4, 16, 64, 64, 64]
        raw_x = x
        x = self.cmt1(x)
        x = self.relu(x + raw_x)

        """第二次下采样"""
        ##flair_2.shape[2, 16, 64, 64, 64]
        t1_1 = x[:, 0]
        t1ce_1 = x[:, 1]
        t2_1 = x[:, 2]
        flair_1 = x[:, 3]
        ##flair_2.shape [2, 32, 32, 32, 32]
        #[1, 16, 64, 64, 64]->[1, 32, 32, 32, 32]
        t1_2 = self.down_2_list[0](t1_1)
        t1ce_2 = self.down_2_list[1](t1ce_1)
        t2_2 = self.down_2_list[2](t2_1)
        flair_2 = self.down_2_list[3](flair_1)
        # [2, 32, 32, 32, 32]...->[2,4, 32, 32, 32, 32]
        x = torch.stack([t1_2, t1ce_2, t2_2, flair_2], dim=1)
        """模态间"""
        # 对四个模态进行融合 x:[2,4, 32, 32, 32, 32]->[2,4, 32, 32, 32, 32]
        raw_x = x
        x = self.cmt2(x)
        x = self.relu(x + raw_x)
        """第三次下采样"""
        #flair_2.shape[2, 32, 32, 32, 32]
        t1_2 = x[:, 0]
        t1ce_2 = x[:, 1]
        t2_2 = x[:, 2]
        flair_2 = x[:, 3]
        ##flair_3.shape [2, 64, 16, 16, 16]
        #[1, 32, 32, 32, 32]->[1, 64, 16, 16, 16]
        t1_3 = self.down_3_list[0](t1_2)
        t1ce_3 = self.down_3_list[1](t1ce_2)
        t2_3 = self.down_3_list[2](t2_2)
        flair_3 = self.down_3_list[3](flair_2)

        # [2, 64, 16, 16, 16]...->[2,4, 64, 16, 16, 16]
        x = torch.stack([t1_3, t1ce_3, t2_3, flair_3], dim=1)
        """模态间"""
        # 对四个模态进行融合 x:[2,4, 64, 16, 16, 16]->[2,4, 64, 16, 16, 16]
        raw_x = x
        x = self.cmt3(x)
        x = self.relu(x + raw_x)

        """第四次下采样"""
        #flair_3.shape [1, 64, 16, 16, 16]
        t1_3 = x[:, 0]
        t1ce_3 = x[:, 1]
        t2_3 = x[:, 2]
        flair_3 = x[:, 3]
        # flair_4.shape [2, 128, 8, 8, 8]
        #[1, 64, 16, 16, 16]->[1, 128, 8, 8, 8]
        t1_4 = self.down_4_list[0](t1_3)
        t1ce_4 = self.down_4_list[1](t1ce_3)
        t2_4 = self.down_4_list[2](t2_3)
        flair_4 = self.down_4_list[3](flair_3)
        x = torch.stack([t1_4,t1ce_4,t2_4,flair_4],dim=1)
        """模态间"""
        #[1, 4,128, 8, 8, 8]->[1, 4,128, 8, 8, 8]
        raw_x = x
        x = self.cmt4(x)
        x = self.relu(x + raw_x)
        t1_4 = x[:, 0]
        t1ce_4 = x[:, 1]
        t2_4 = x[:, 2]
        flair_4 = x[:, 3]

        out = []
        t1_tuple = (t1_4, t1_3, t1_2, t1_1, t1_0)
        out.append(t1_tuple)
        t1ce_tuple = (t1ce_4, t1ce_3, t1ce_2, t1ce_1, t1ce_0)
        out.append(t1ce_tuple)
        t2_tuple = (t2_4, t2_3, t2_2, t2_1, t2_0)
        out.append(t2_tuple)
        flair_tuple = (flair_4, flair_3, flair_2, flair_1, flair_0)
        out.append(flair_tuple)

        return out


class ModelEncoder(nn.Module):

    def __init__(self, model_num,
                 img_size,
                 fea,
                 pool_size,
                 ):

        super().__init__()
        self.model_num = model_num
        self.encoder = EncoderLayer(img_size=img_size,in_channels=1,pool_size=pool_size,features=fea)#后加

    def forward(self, x):
        x = x.unsqueeze(dim=2)  # x的维度为[1,4,1,128,128,128],将通道数扩展为1
        encoder_out = self.encoder(x)
        """
        encoder_out是一个列表，列表有四个元素(模态)，每个元素是一个元组(x4, x3, x2, x1, x0) 
        x4, x3, x2, x1, x0 是下采样过程中的特征
        """
        return encoder_out