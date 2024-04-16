import torch.nn as nn
from .encoder.Encoder import ModelEncoder, UpCat
from .fusion.BottleNeck import ModelBottleNeck
import torch
from einops import rearrange

class CMITNet(nn.Module):
    def __init__(self, model_num,
                 out_channels,
                 image_size,
                 window_size,
                 fea = (16, 16, 32, 64, 128, 16),
                 pool_size=((2,2,2), (2,2,2), (2,2,2), (2,2,2)),
                 cross_num_layer=2,
                 ):

        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.pool_size = pool_size

        pool_size_all = [1, 1, 1]
        image_size_s = [image_size]
        for p in pool_size:
            pool_size_all = [pool_size_all[i] * p[i] for i in range(len(p))]

            image_size_s.append((image_size_s[-1][0] // p[0], image_size_s[-1][1] // p[1], image_size_s[-1][2] // p[2]))
        new_image_size = [image_size[i] // pool_size_all[i] for i in range(3)]

        self.encoder = ModelEncoder(model_num=model_num,
                                                img_size=image_size_s[1:],
                                                fea=fea, pool_size=pool_size)

        self.Ttd = ModelBottleNeck(model_num=model_num,
                                in_channels=fea[4],
                                hidden_size=fea[4],
                                img_size=new_image_size,
                                cross_num_layer=cross_num_layer,
                                mlp_size=2*fea[4],
                                window_size=window_size)


        self.upcat_4 = UpCat(fea[4], fea[3], fea[3], pool_size=pool_size[3], is_last=False)
        self.upcat_3 = UpCat(fea[3], fea[2], fea[2], pool_size=pool_size[2], is_last=False)
        self.upcat_2 = UpCat(fea[2], fea[1], fea[1], pool_size=pool_size[1], is_last=False)
        self.upcat_1 = UpCat(fea[1], fea[0], fea[5], pool_size=pool_size[0], is_last=True)

        self.relu = nn.ReLU()
        self.final_conv = nn.Conv3d(fea[5], out_channels, 1, 1)

    def forward(self, x):
        assert x.shape[1] == self.model_num

        encoder_x = self.encoder(x)

        # encoder_1.shape [2, 4, 16, 128, 128, 128]
        encoder_1 = torch.stack([encoder_x[i][4] for i in range(self.model_num)], dim=1)
        #encoder_2.shape [2, 4, 16, 64, 64, 64]
        encoder_2 = torch.stack([encoder_x[i][3] for i in range(self.model_num)], dim=1)
        #encoder_3.shape [2, 4, 32, 32, 32, 32]
        encoder_3 = torch.stack([encoder_x[i][2] for i in range(self.model_num)], dim=1)
        #encoder_4.shape [2, 4, 64, 16, 16, 16]
        encoder_4 = torch.stack([encoder_x[i][1] for i in range(self.model_num)], dim=1)
        #encoder_5.shape [2, 4, 128, 8, 8, 8]
        encoder_5 = torch.stack([encoder_x[i][0] for i in range(self.model_num)], dim=1)
        #模态内
        fusion_out = self.Ttd(encoder_5)

        # 模态间已经融合了
        # [1, 128, 8, 8, 8]
        fusion_out_ = self.relu(encoder_5[:, 0] + encoder_5[:, 1] + encoder_5[:, 2] + encoder_5[:, 3])
        fusion_out = fusion_out + fusion_out_

        encoder_1_ = self.relu(encoder_1[:, 0] + encoder_1[:, 1] + encoder_1[:, 2] + encoder_1[:, 3])
        encoder_2_ = self.relu(encoder_2[:, 0] + encoder_2[:, 1] + encoder_2[:, 2] + encoder_2[:, 3])
        encoder_3_ = self.relu(encoder_3[:, 0] + encoder_3[:, 1] + encoder_3[:, 2] + encoder_3[:, 3])
        encoder_4_ = self.relu(encoder_4[:, 0] + encoder_4[:, 1] + encoder_4[:, 2] + encoder_4[:, 3])

        # [1, 128, 8, 8, 8] [1, 64, 16, 16, 16]->[1, 64, 16, 16, 16]
        u4 = self.upcat_4(fusion_out, encoder_4_)
        # [1, 64, 16, 16, 16] [1, 32, 32, 32, 32]->[1, 32, 32, 32, 32]
        u3 = self.upcat_3(u4, encoder_3_)
        # [1, 32, 32, 32, 32] [1, 16, 64, 64, 64]->[1, 16, 64, 64, 64]
        u2 = self.upcat_2(u3, encoder_2_)
        # [1, 16, 64, 64, 64] [1, 16, 128, 128, 128]->[1, 16, 128, 128, 128]
        u1 = self.upcat_1(u2, encoder_1_)

        logits = self.final_conv(u1)

        return logits
