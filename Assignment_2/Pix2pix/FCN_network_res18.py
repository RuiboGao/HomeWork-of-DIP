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
        # ResNet-18 根据ResNet-18 编写的解码器 前向传播将高和宽缩小为原来的1/32
        self.conv0 = nn.Sequential(nn.Conv2d(3,64,kernel_size=6,stride=2,padding=3),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3,stride=1,padding=1))
        
        self.conv1 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                   nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.R1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                   nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.R2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                   nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.en1 = nn.Sequential(nn.Conv2d(64,128,kernel_size=1,stride=2,padding=0))
        self.R3 = nn.ReLU(inplace=False)

        self.conv4 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                   nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.R4 = nn.ReLU(inplace=False)

        self.conv5 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                   nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.R5 = nn.ReLU(inplace=False)
        self.en2 = nn.Sequential(nn.Conv2d(128,256,kernel_size=1,stride=2,padding=0))

        self.conv6 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                   nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.R6 = nn.ReLU(inplace=False)

        # self.conv7 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
        #                            nn.BatchNorm2d(512),nn.ReLU(),
        #                            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        #                            nn.BatchNorm2d(512),nn.ReLU())
        # self.R7 = nn.ReLU()
        # self.en3 = nn.Sequential(nn.Conv2d(256,512,kernel_size=1,stride=2,padding=0))
        
        # self.conv8 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        #                            nn.BatchNorm2d(512),nn.ReLU(),
        #                            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        #                            nn.BatchNorm2d(512),nn.ReLU())
        # self.R8 = nn.ReLU()
		
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        # 因为ResNet-18 将原图片缩小为原来的1/32，因此解码器需要将卷积层输出结果放大32倍
        # 根据反卷积的特点，将填充设为16，核的高和宽设为64，即可将输入放大s倍，因此取步长32
        # 因此可取k_h=k_w=64,padding=16,strid=32,并且需要将输出通道数设置为数据集的类数num_classes
        # self.final_conv = nn.Sequential(nn.Conv2d(512,num_classes,kernel_size=1))
        
        self.transpose_conv1 = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1),
                                             nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                             nn.ConvTranspose2d(128,128,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.R_t1 = nn.ReLU(inplace=False)
        
        self.transpose_conv2 = nn.Sequential(nn.ConvTranspose2d(128,128,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                             nn.ConvTranspose2d(128,128,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        
        self.R_t2 = nn.ReLU(inplace=False)
        
        self.transpose_conv3 = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1),
                                             nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                             nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        
        self.R_t3 = nn.ReLU(inplace=False)
        
        self.transpose_conv4 = nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                             nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.R_t4 = nn.ReLU(inplace=False)
        
        self.outconv = nn.Sequential(nn.ConvTranspose2d(64,3,kernel_size=6,stride=2,padding=3))
        


    def forward(self, x):
        # Encoder forward pass 
        # ResNet-18
        # Decoder forward pass
        x = self.conv0(x)

        x = self.R1(self.conv1(x)+x)

        x = self.R2(self.conv2(x)+x)

        x = self.R3(self.conv3(x)+self.en1(x))
        
        x = self.R4(self.conv4(x)+x)
   
        x = self.R5(self.conv5(x)+self.en2(x))

        x = self.R6(self.conv6(x)+x)

        # x = self.conv7(x)
        # x = x + self.en3(lay6)
        # x = self.R7(x)
        # lay7 = x

        # x = self.conv8(x)
        # x = x + lay7
        # x = self.R8(x)
        # lay8 = x

        ### FILL: encoder-decoder forward pass

        #Size of input=1,num_classes,256,256

        x = self.transpose_conv1(x)
        # x = x + lay4
        x = self.R_t1(x)

        x = self.transpose_conv2(x)
        # x = x + lay3
        x = self.R_t2(x)

        x = self.transpose_conv3(x)
        # x = x + lay2
        x = self.R_t3(x)

        x = self.transpose_conv4(x)
        # x = x + lay1
        x = self.R_t4(x)
        
        x = self.outconv(x)

        return x
    