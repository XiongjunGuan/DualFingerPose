import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .CBAM import CBAM
from .resnext import ResNextBlock



class NormalizeModule(nn.Module):

    def __init__(self, m0=0, var0=1, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m)**2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


class ConvBnPRelu(nn.Module):

    def __init__(self,
                 in_chn,
                 out_chn,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chn,
                              out_chn,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chn, eps=0.001, momentum=0.99)
        self.relu = nn.PReLU(out_chn, init=0)

    def forward(self, input):
        y = self.conv(input)
        y = self.bn(y)
        y = self.relu(y)
        return y


class DualFingerPose(nn.Module):

    def __init__(
        self,
        inp_mode="patch_cap",
        trans_out_form="claSum",
        trans_num_classes=512,
        rot_out_form="claSum",
        rot_num_classes=120,
        main_channel_lst=[64, 128, 256, 512, 1024],
        layer_lst=[3, 4, 6, 3],
        aux_channel_lst=[32, 64, 128, 256, 512],
    ):
        super(DualFingerPose, self).__init__()
        assert inp_mode=="patch_cap"
        assert trans_out_form=="claSum"
        assert rot_out_form=="claSum"

        self.trans_out_form = trans_out_form
        self.rot_out_form = rot_out_form

        self.norm_layer = NormalizeModule(m0=0, var0=1)

        self.main_layer1 = nn.Sequential(
            ConvBnPRelu(1, main_channel_lst[0], 7, stride=2, padding=3),
            ConvBnPRelu(main_channel_lst[0], main_channel_lst[0], 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
  
        self.aux_layer1 = nn.Sequential(
            ConvBnPRelu(1, aux_channel_lst[0], 3),
            ConvBnPRelu(aux_channel_lst[0], aux_channel_lst[0], 3))
    

        self.main_layer2 = self._make_layers(ResNextBlock,
                                             in_channels=main_channel_lst[0],
                                             out_channels=main_channel_lst[1],
                                             groups=32,
                                             stride=2,
                                             num_layers=layer_lst[0])
        self.aux_layer2 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[0],
            out_channels=aux_channel_lst[1],
            groups=32,
            stride=1 ,
            num_layers=layer_lst[0])

        self.main_layer3 = self._make_layers(ResNextBlock,
                                             in_channels=main_channel_lst[1],
                                             out_channels=main_channel_lst[2],
                                             groups=32,
                                             stride=2,
                                             num_layers=layer_lst[1])
        self.main_att3 = CBAM(main_channel_lst[2])

        self.aux_layer3 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[1],
            out_channels=aux_channel_lst[2],
            groups=32,
            stride=1,
            num_layers=layer_lst[1])
        self.aux_att3 = CBAM(aux_channel_lst[2])

        self.main_layer4 = self._make_layers(ResNextBlock,
                                             in_channels=main_channel_lst[2],
                                             out_channels=main_channel_lst[3],
                                             groups=32,
                                             stride=2,
                                             num_layers=layer_lst[2])
        self.main_att4 = CBAM(main_channel_lst[3])

        self.aux_layer4 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[2],
            out_channels=aux_channel_lst[3],
            groups=32,
            stride=1 ,
            num_layers=layer_lst[2])
        self.aux_att4 = CBAM(aux_channel_lst[3])

        self.main_layer5 = self._make_layers(ResNextBlock,
                                             in_channels=main_channel_lst[3],
                                             out_channels=main_channel_lst[4],
                                             groups=32,
                                             stride=2,
                                             num_layers=layer_lst[3])
        self.main_att5 = CBAM(main_channel_lst[4])

        self.aux_layer5 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[3],
            out_channels=aux_channel_lst[4],
            groups=32,
            stride=1 ,
            num_layers=layer_lst[3])
        self.aux_att5 = CBAM(aux_channel_lst[4])

        self.avgpool_flatten_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                   nn.Flatten(start_dim=1))
        
        self.trans_fc_theta = nn.Linear(
            main_channel_lst[4] + aux_channel_lst[4], trans_num_classes)
       
        
        self.rot_fc_theta = nn.Linear(
            main_channel_lst[4] + aux_channel_lst[4], rot_num_classes)
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, Block, in_channels, out_channels, groups, stride,
                     num_layers):
        layers = []
        layers.append(
            Block(in_channels=in_channels,
                  out_channels=out_channels,
                  groups=groups,
                  stride=stride))
        for _ in range(num_layers - 1):
            layers.append(
                Block(in_channels=out_channels,
                      out_channels=out_channels,
                      groups=groups,
                      stride=1))
        return nn.Sequential(*layers)

    def forward(self, inp):
        [main_inp, aux_inp] = inp
        main_inp = self.norm_layer(main_inp)
        main_feat = self.main_layer1(main_inp)
        main_feat = self.main_layer2(main_feat)
        main_feat = self.main_layer3(main_feat)
        main_feat, _, _ = self.main_att3(main_feat)
        main_feat = self.main_layer4(main_feat)
        main_feat, _, _ = self.main_att4(main_feat)
        main_feat = self.main_layer5(main_feat)
        main_feat, _, _ = self.main_att5(main_feat)

        aux_feat = self.aux_layer1(aux_inp)
        aux_feat = self.aux_layer2(aux_feat)
        aux_feat = self.aux_layer3(aux_feat)
        aux_feat, _, _ = self.aux_att3(aux_feat)
        aux_feat = self.aux_layer4(aux_feat)
        aux_feat, _, _ = self.aux_att4(aux_feat)
        aux_feat = self.aux_layer5(aux_feat)
        aux_feat, _, _ = self.aux_att5(aux_feat)

        main_feat = self.avgpool_flatten_layer(main_feat)
        aux_feat = self.avgpool_flatten_layer(aux_feat)
        feat = torch.cat([main_feat, aux_feat], dim=1)

        # --- translation
        pred_xy = self.trans_fc_theta(feat)
        _, c = pred_xy.shape
        pred_x = F.softmax(pred_xy[:, :c // 2], dim=1)
        pred_y = F.softmax(pred_xy[:, c // 2:], dim=1)
        pred_xy = torch.cat(
            [pred_x, pred_y],
            dim=1)  # [b, (num_class//2,num_class//2)] for x and y prob
       
        # --- rotation
        pred_theta = self.rot_fc_theta(feat)
        pred_theta = F.softmax(pred_theta,
                                dim=1)  # [b, num_class] for theta prob
      

        return [pred_xy, pred_theta]



