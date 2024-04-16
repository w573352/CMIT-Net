import torch.nn as nn
from .Ttd import ModelTtd

class ModelBottleNeck(nn.Module):
    def __init__(self, model_num,
                 in_channels,
                 hidden_size,
                 img_size,
                 cross_num_layer,#表示用了几层融合
                 mlp_size,
                 window_size):
        super().__init__()
        """
        model_num  4
        in_channels  128
        hidden_size  128
        img_size  [8, 8, 8]
        mlp_size  256
        self_num_layer  2
        window_size  (4, 4, 4)
        cross_num_layer 4
        """
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.model_num = model_num
        self.Ttd1 = nn.ModuleList([ModelTtd(in_channels=in_channels,
                                                hidden_size=hidden_size,
                                                img_size=img_size,
                                                is_position=True,
                                                mlp_size=mlp_size,
                                                window_size=window_size) for i in range(model_num)])

        self.Ttd2 = nn.ModuleList([ModelTtd(in_channels=in_channels,
                                                                   hidden_size=hidden_size,
                                                                   img_size=img_size,
                                                                   is_position=True,
                                                                   mlp_size=mlp_size,
                                                                   window_size=window_size) for i in range(model_num)])

        self.relu = nn.ReLU()
    def forward(self, x):
        # x: (batch, modal_num, hidden_size, d, w, h) x即为编码器输出的高级特征
        # x的维度为[1, 4, 128, 8, 8, 8] 4是模态数 128是通道数
        #四个模态数据分别tri-attention + 残差收缩单元块x:[1, 4, 128, 8, 8, 8]->[1, 4, 128, 8, 8, 8]
        """第一层"""
        for i in range(self.model_num):#self.model_num = 4
            x_i = x[:, i].clone()  # 创建张量的副本
            x_i = self.Ttd1[i](x_i)  # 对副本进行操作
            #x[:,i].shape [2, 128, 8, 8, 8]
            x[:,i] = x_i

        """第二层"""
        #四个模态数据分别tri-attention + 残差收缩单元块x:[1, 4, 128, 8, 8, 8]->[1, 4, 128, 8, 8, 8]
        for i in range(self.model_num):
            x_i = x[:, i].clone()  # 创建张量的副本
            x_i = self.Ttd2[i](x_i)  # 对副本进行操作
            # x[:,i].shape [2, 128, 8, 8, 8]
            x[:, i] = x_i

        t1 = x[:, 0]
        t1ce = x[:, 1]
        t2 = x[:, 2]
        flair = x[:, 3]
        fusion_out = self.relu(flair + t1 + t1ce + t2)
        return fusion_out#([1, 128, 8, 8, 8])
