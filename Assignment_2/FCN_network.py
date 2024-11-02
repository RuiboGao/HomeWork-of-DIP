import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8 卷积核个数 卷积核的输入通道数由输入矩阵的通道数决定，输出矩阵由卷积核
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True)  
        # )
        ### FILL: add more CONV Layers
        # 根据U-Net 编写的解码器 
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=0),
                                     nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                     nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0),
                                     nn.BatchNorm2d(64),nn.ReLU(inplace=True))

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2,stride=2) # 最大池化大小减半，通道不变

        self.conv2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                   nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2,stride=2) # 最大池化大小减半，通道不变

        self.conv3 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                   nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2,stride=2) # 最大池化大小减半，通道不变

        self.conv4 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(512),nn.ReLU(inplace=True),
                                   nn.Conv2d(512,512,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2,stride=2) # 最大池化大小减半，通道不变

        self.conv5 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(1024),nn.ReLU(inplace=True),
                                   nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(1024),nn.ReLU(inplace=True))
        
        # Decoder (Deconvolutional Layers)
        # U-Net的解码器
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.up_1 = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))

        self.conv6 = nn.Sequential(nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(512),nn.ReLU(inplace=True),
                                   nn.Conv2d(512,512,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        
        self.up_2 = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))

        self.conv7 = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                   nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        
        self.up_3 = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))

        self.conv8 = nn.Sequential(nn.Conv2d(256,128,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                   nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        
        self.up_4 = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))

        self.conv9 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                   nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        
        self.conv10 = nn.Sequential(nn.Conv2d(64,2,kernel_size=1,stride=1,padding=0))


    def forward(self, x):
        # Encoder forward pass 
        # U-Net
        # Decoder forward pass
        ### FILL: encoder-decoder forward pass
        print(type(x))
        x = self.conv1(x)
        lay1 = x

        x = self.max_pool_1(x)
        x = self.conv2(x)
        lay2 = x

        x = self.max_pool_2(x)
        x = self.conv3(x)
        lay3 = x

        x = self.max_pool_3(x)
        x = self.conv4(x)
        lay4 = x

        x = self.max_pool_4(x)
        x = self.conv5(x)
        lay5 = x

        x = self.up_1(x)
        diffY = lay4.size()[2] - x.size()[2]
        diffX = lay4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = nn.torch.cat([lay4, x], dim=1)#进行融合裁剪
        x = self.conv6(x)
        lay6 = x

        x = self.up_2(x)
        diffY = lay3.size()[2] - x.size()[2]
        diffX = lay3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = nn.torch.cat([lay3, x], dim=1)#进行融合裁剪
        x = self.conv7(x)
        lay7 = x

        x = self.up_3(x)
        diffY = lay2.size()[2] - x.size()[2]
        diffX = lay2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = nn.torch.cat([lay2, x], dim=1)#进行融合裁剪
        x = self.conv8(x)
        lay8 = x

        x = self.up_4(x)
        diffY = lay1.size()[2] - x.size()[2]
        diffX = lay1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = nn.torch.cat([lay1, x], dim=1)#进行融合裁剪
        x = self.conv9(x)
        # lay9 = x

        output = self.conv10(x)
        return output
    