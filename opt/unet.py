from icecream import ic

from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F

import pdb


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class VGG16UNet(nn.Module):
    def __init__(self, in_channel=4, out_channel=3, num_filters=32, pretrained=True, requires_grad=True,args=None):
        """
        out_channel:输出图像的通道数
        pretrained: 是否加载预训练模型
        """
        super().__init__()
        self.out_channel = out_channel

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features#

        self.relu = nn.ReLU(inplace=True)

        self.args = args

        # Fix vgg16
        # if not requires_grad: 
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False


    
        self.encoder_conv1_1=nn.Conv2d(in_channel, 64, 3, padding=(1, 1))#
        self.conv1 = nn.Sequential(self.encoder_conv1_1,
                                self.relu,
                                self.encoder[2],
                                self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        # self.conv5 = nn.Sequential(self.encoder[24],
        #                            self.relu,
        #                            self.encoder[26],
        #                            self.relu,
        #                            self.encoder[28],
        #                            self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        # self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, out_channel, kernel_size=1)

        self.c_pool = nn.AdaptiveMaxPool2d((2,2))
        self.classifer_fc = nn.Sequential(nn.Linear(512*2*2,(512*2*2)//8),
                                          self.relu,
                                          nn.Linear((512*2*2)//8,1)
                                          )

    def forward(self, x, c, return_c=False, cmask=False):
        # assert cmask==True

        # input_shape =x.shape
        # assert(len(input_shape) >=3)
        # if len(input_shape) ==3:
        #     x = x.unsqueeze(0)
        # if input_shape[-1]==1 or input_shape[-1]==3:
        #     x = x.permute(0,3,1,2)

        # print(x.shape)
        assert x.shape[2] == 756 or x.shape[2] == 800
        assert x.shape[3] == 1008 or x.shape[2] == 800
        # x = F.interpolate(x,size=(752,1008))
        x = F.interpolate(x,size=(self.args.wm_resize,self.args.wm_resize))
        # c = F.interpolate(c,size=(self.args.wm_resize,self.args.wm_resize))

        
        c = c.unsqueeze(-1).unsqueeze(-1)
        c = F.interpolate(c,size=(self.args.wm_resize,self.args.wm_resize))
        x = torch.cat( (x,c), 1)

        conv1 = self.conv1(x) # torch.Size([2, 64, 756, 1008])
        conv2 = self.conv2(self.pool(conv1)) # torch.Size([2, 128, 378, 504])
        conv3 = self.conv3(self.pool(conv2)) # torch.Size([2, 256, 189, 252])
        conv4 = self.conv4(self.pool(conv3)) # torch.Size([2, 512, 94, 126])
        # conv5 = self.conv5(self.pool(conv4))

        # center = self.center(self.pool(conv5))\
        center = self.center(self.pool(conv4)) # torch.Size([2, 256, 94, 126]) center <> upsampling module 
        # pool(conv4) 1,512,3,3

        # x_c = self.c_pool(conv4) #1,512,3,3
        # x_c = F.sigmoid(self.classifer_fc(x_c.view(x_c.shape[0],-1)))

        # dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([center, conv4], 1)) # torch.Size([2, 256, 188, 252])

        # dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        # pdb.set_trace()

        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        
        # x_out=F.relu(self.final(dec1))

        x_out=self.final(dec1)
        
        # x_out=F.sigmoid(self.final(dec1))

        # if cmask:
        #     x_out = x_out*x_c.unsqueeze(-1).unsqueeze(-1)
        
        # x_out = F.interpolate(x_out,size=(756,1008))

        # x_out = x_out.permute(0,2,3,1)
        # if len(input_shape)==3:
        #     x_out = x_out.squeeze()

        # if return_c:
        #     return x_out, x_c
        # else:
        
        return x_out



class Classifier(nn.Module):
    def __init__(self, out_channel=3, num_filters=32, pretrained=True, requires_grad=True):
        super().__init__()
        self.out_channel = out_channel

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features#
        self.relu = nn.ReLU(inplace=True)

        # Fix vgg16
        # if not requires_grad: 
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False

        # self.encoder_conv1_1=nn.Conv2d(out_channel, 64, 3, padding=(1, 1))#
        '''
        self.conv1 = nn.Sequential(self.encoder_conv1_1,
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)
        '''
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        # self.conv5 = nn.Sequential(self.encoder[24],
        #                            self.relu,
        #                            self.encoder[26],
        #                            self.relu,
        #                            self.encoder[28],
        #                            self.relu)

        # self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        # # self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        # self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        # self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        # self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        # self.dec1 = ConvRelu(64 + num_filters, num_filters)
        # self.final = nn.Conv2d(num_filters, out_channel, kernel_size=1)

        self.c_pool = nn.AdaptiveMaxPool2d((2,2))
        self.classifer_fc = nn.Sequential(nn.Linear(512*2*2,(512*2*2)//8),
                                          self.relu,
                                          nn.Linear((512*2*2)//8,1)
                                          )

    def forward(self, x, return_c=False, cmask=False):
        # assert cmask==True

        # input_shape =x.shape
        # assert(len(input_shape) >=3)
        # if len(input_shape) ==3:
        #     x = x.unsqueeze(0)
        # if input_shape[-1]==1 or input_shape[-1]==3:
        #     x = x.permute(0,3,1,2)

        # print(x.shape)
        assert x.shape[2] == 756 or x.shape[2] == 800
        assert x.shape[3] == 1008 or x.shape[2] == 800
        # x = F.interpolate(x,size=(752,1008))
        x = F.interpolate(x,size=(512,512))

        conv1 = self.conv1(x) # torch.Size([2, 64, 756, 1008])
        conv2 = self.conv2(self.pool(conv1)) # torch.Size([2, 128, 378, 504])
        conv3 = self.conv3(self.pool(conv2)) # torch.Size([2, 256, 189, 252])
        conv4 = self.conv4(self.pool(conv3)) # torch.Size([2, 512, 94, 126])
        # conv5 = self.conv5(self.pool(conv4))

        # center = self.center(self.pool(conv5))\
        # center = self.center(self.pool(conv4)) # torch.Size([2, 256, 94, 126]) center <> upsampling module 
        # pool(conv4) 1,512,3,3

        x_c = self.c_pool(conv4) #1,512,3,3
        x_c = F.sigmoid(self.classifer_fc(x_c.view(x_c.shape[0],-1)))

        # dec5 = self.dec5(torch.cat([center, conv5], 1))
        # dec4 = self.dec4(torch.cat([center, conv4], 1)) # torch.Size([2, 256, 188, 252])

        # dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        # pdb.set_trace()

        # dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        # dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        # dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        
        # x_out=F.relu(self.final(dec1))

        # x_out=self.final(dec1)
        
        # x_out=F.sigmoid(self.final(dec1))

        # if cmask:
        #     x_out = x_out*x_c.unsqueeze(-1).unsqueeze(-1)
        
        # x_out = F.interpolate(x_out,size=(756,1008))

        # x_out = x_out.permute(0,2,3,1)
        # if len(input_shape)==3:
        #     x_out = x_out.squeeze()
        return x_c
        # if return_c:
        #     return x_out, x_c
        # else:
        #     return x_out
        