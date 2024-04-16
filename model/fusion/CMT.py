
import math
from einops import rearrange
import torch.nn as nn
import torch
from .layers import get_config, Mlp,Cross_Mlp
import torch.nn.functional as F

class CCDPA(nn.Module):
    def __init__(self,config):
        super(CCDPA, self).__init__()
        self.to_query = nn.Linear(config.img_dim, config.img_dim)
        self.to_key = nn.Linear(config.img_dim, config.img_dim)
        self.conv_list = nn.ModuleList([nn.Conv3d(in_channels=config.in_channels,  # 128
                                          out_channels=config.in_channels,  # 128
                                          kernel_size=(1, 1, 1),  # (1, 1, 1)
                                          stride=(1, 1, 1)) for i in range(config.model_num)])
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Conv3d(in_channels=config.in_channels,  # 128
                                          out_channels=config.in_channels,  # 128
                                          kernel_size=(1, 1, 1),  # (1, 1, 1)
                                          stride=(1, 1, 1))
    def forward(self, m1,m2,m3,m4):
        """
        m2.shape m1 m2 m3 m4 [1, 128, 8, 8, 8]
        将m2 m3 m4 融合进m1
        """
        B,C,D,W,H = m1.shape
        #m1.shape [1, 128, 8, 8, 8] m1Q.shape [1, 128, 1, 1, 8]
        m1Q = F.adaptive_avg_pool3d(m1, (D, 1, 1)).permute(0, 1, 3, 4, 2)
        #m1Q.shape [1, 128, 1, 1, 8]
        m1Q = self.to_query(m1Q)

        #m1K.shape [1, 128, 1, 1, 8]
        m1K = F.adaptive_avg_pool3d(m1, (D, 1, 1)).permute(0, 1, 3, 4, 2)
        m2K = F.adaptive_avg_pool3d(m2, (D, 1, 1)).permute(0, 1, 3, 4, 2)
        m3K = F.adaptive_avg_pool3d(m3, (D, 1, 1)).permute(0, 1, 3, 4, 2)
        m4K = F.adaptive_avg_pool3d(m4, (D, 1, 1)).permute(0, 1, 3, 4, 2)
        # key.shape [1, 128, 1, 4, 8]
        key = torch.cat((m1K,m2K,m3K,m4K),3)
        #[2, 128, 1, 4, 8]
        key = self.to_key(key)
        #m1K.shape [1, 128, 1, 1, 8]
        m1K,m2K,m3K,m4K = key.chunk(4,3)

        #[1, 128, 1, 1, 8] [1, 128, 1, 8, 1]->[1, 128, 1, 1, 1]
        a1 = torch.matmul(m1Q,m1K.transpose(-2, -1))
        a2 = torch.matmul(m1Q, m2K.transpose(-2, -1))
        a3 = torch.matmul(m1Q, m3K.transpose(-2, -1))
        a4 = torch.matmul(m1Q,m4K.transpose(-2, -1))
        #[1, 128, 1, 1, 4]
        a = torch.cat((a1,a2,a3,a4),-1)
        a = a / math.sqrt(D)#[1, 128, 1, 1, 4]
        a = self.softmax(a)#[1, 128, 1, 1, 4]

        #a1.shape [1, 128, 1, 1, 1]
        a1,a2,a3,a4 = a.chunk(4,-1)

        #m1.shape [1, 128, 8, 8, 8]
        m1 = self.conv_list[0](m1)
        m2 = self.conv_list[1](m2)
        m3 = self.conv_list[2](m3)
        m4 = self.conv_list[3](m4)

        #m1.shape [1, 128, 8, 8, 8]
        m1 = a1 * m1
        m2 = a2 * m2
        m3 = a3 * m3
        m4 = a4 * m4
        m = m1 + m2 + m3 + m4
        #m.shape [2, 128, 8, 8, 8]
        return m

class CMSA(nn.Module):
    def __init__(self,config):
        super(CMSA, self).__init__()
        self.ccdpa1 = CCDPA(config)
        self.ccdpa2 = CCDPA(config)
        self.ccdpa3 = CCDPA(config)
        self.ccdpa4 = CCDPA(config)
    def forward(self, t1,t1ce,t2,flair):
        """
        整个多模态融合模块
        flair.shape [2, 128, 8, 8, 8]
        """
        #flair t1 t1ce t2.shape [2, 128, 8, 8, 8]
        t1 = self.ccdpa1(t1,t1ce,t2,flair)
        t1ce = self.ccdpa2(t1ce,t1,t2,flair)
        t2 = self.ccdpa3(t2,t1,t1ce,flair)
        flair = self.ccdpa4(flair,t1,t1ce,t2)


        return t1, t1ce, t2, flair

class ModelCMTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config#config.model_num 4
        self.model_num = config.model_num
        self.img_size = config.img_size
        self.layer_norm_list1 = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=1e-6) for i in range(config.model_num)])
        self.layer_norm_list2 = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=1e-6) for i in range(config.model_num)])
        self.mlp_list = nn.ModuleList([Cross_Mlp(config) for i in range(config.model_num)])
        self.cmsa = CMSA(config)

    def forward(self,x):#x.shape[1, 4, 128, 8, 8, 8] 归一化 + 融合 + 归一化 + MLP
        B, M, C, D, W, H = x.shape
        raw1_x = x
        """归一化"""
        #改编x的维度进行归一化[1, 4, 128, 8, 8, 8]->[1, 4, 512,128]
        x = rearrange(x,"b m c d w h -> b m (d w h) c")
        # 归一化[1, 4, 512,128]->[1, 4, 512,128]
        for i in range(self.config.model_num):
            #[1,512,128] - >[1,512,128]
            x_i = x[:,i].clone()
            x_i = self.layer_norm_list1[i](x_i)
            x[:,i] = x_i

        """融合"""
        #改编x的维度来进行融合[1, 4, 512,128]->[1, 4, 128, 8, 8, 8]
        x = rearrange(x,"b m (d w h) c -> b m c d w h",d=D,w=W,h=H)
        t1 = x[:, 0]
        t1ce = x[:, 1]
        t2 = x[:, 2]
        flair = x[:, 3]
        #flair, t1, t1ce, t2.shape [2, 128, 8, 8, 8]
        t1, t1ce, t2, flair = self.cmsa(t1, t1ce, t2, flair)  # 整个大的融合模块

        #[1, 128, 8, 8, 8]->[1, 4, 128, 8, 8, 8]
        x = torch.stack((t1, t1ce, t2, flair),dim=1)
        """残差连接"""
        #[1, 4, 128, 8, 8, 8]->[1, 4, 128, 8, 8, 8]
        x = x + raw1_x

        #raw2_x.shape [2, 4, 128, 8, 8, 8]
        raw2_x = x
        """归一化"""
        # 改编x的维度进行归一化[1, 4, 128, 8, 8, 8]->[1, 4, 512,128]
        x = rearrange(x, "b m c d w h -> b m (d w h) c")
        # 归一化[1, 4, 512,128]->[1, 4, 512,128]
        for i in range(self.config.model_num):
            # [1,512,128] - >[1,512,128]
            x_i = x[:, i].clone()
            x_i = self.layer_norm_list2[i](x_i)
            x[:, i] = x_i
        """MLP + 残差连接"""
        #[2, 4, 512, 128] - > [2, 4, 128, 8, 8, 8]
        x = rearrange(x,"b m (d w h) c -> b m c d w h",d=D,w=W,h=H)
        # [1, 4, 128, 8, 8, 8] - >[1, 4, 128, 8, 8, 8]
        for i in range(self.config.model_num):
            x_i = x[:, i].clone()
            #[2, 128, 8, 8, 8] - > [2, 128, 8, 8, 8]
            x_i = self.mlp_list[i](x_i)
            x[:, i] = x_i
        x = x + raw2_x

        return x#[1, 4, 128, 8, 8, 8]


class ModelCMT(nn.Module):
    """
        model_num: 4 in_channels: 128 hidden_size: 128
        img_size: [8, 8, 8] mlp_size:256 token_mixer_size:32
        token_learner=True
    """
    def __init__(self, model_num, in_channels,
                 hidden_size,
                 img_size, img_dim,mlp_size=256):
        super().__init__()
        self.embeddings = nn.ModuleList([])
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        patch_size = (1, 1, 1)
        self.config = get_config(in_channels=in_channels, hidden_size=hidden_size, patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size,model_num=model_num,img_dim=img_dim)
        self.model_num = model_num
        self.img_size = img_size

        self.model_cmt = ModelCMTBlock(config=self.config)

    def forward(self,x):#x.shape [1, 4, 128, 8, 8, 8]
        B,M,C,D,W,H = x.shape
        """corss_atttention的输出形式是token的形式，因为MLP和归一化要求token维度输入"""
        # [1, 4, 128, 8, 8, 8]->[1, 4, 128, 8, 8, 8]
        x = self.model_cmt(x)
        return x#[1, 4, 128, 8, 8, 8]

