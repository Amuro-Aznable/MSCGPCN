#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    定义了CMLP、编码器、生成器、鉴别器等网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import extract_outline


class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128),2)
        x_256 = torch.squeeze(self.maxpool(x_256),2)
        x_512 = torch.squeeze(self.maxpool(x_512),2)
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        return x

# PF-Net论文里的CMLP实现了一个多尺度的特征提取过程，每个尺度通过不同的 Convlayer 模块处理输入数据，并最终将各尺度处理的特征拼接并加工成最终的输出
class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)       
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs,2)
        latentfeature = latentfeature.transpose(1,2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature,1)
#        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))  
#        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
#        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
#        latentfeature = latentfeature + latentfeature_64
#        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
#        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
#        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
#        latentfeature = latentfeature + latentfeature_256
#        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
#        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
#        latentfeature = self.maxpool(latentfeature)
#        latentfeature = torch.squeeze(latentfeature,2)
        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list,k=40):
        super(PointcloudCls,self).__init__()
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class _netG(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(_netG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        
        self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(128,6,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
      
    def forward(self,x):
        x = self.latentfeature(x) # 通过CMLP提取了(batch,1920)的特征
        x_1 = F.relu(self.fc1(x)) # 经过线性层得到(batch,1024)特征
        x_2 = F.relu(self.fc2(x_1)) # 经过线性层得到(batch,512)特征
        x_3 = F.relu(self.fc3(x_2))  # 经过线性层得到(batch,256)特征
        
        # 数字均以main()的示例为参考，以下为论文中的PPD解码操作,推测出M1=64,M2=128,M=512(main()示例的M为1024)
        pc1_feat = self.fc3_1(x_3) # (batch,192),192=3x64对应论文中PPD解码器对FC3的3M1
        pc1_xyz = pc1_feat.reshape(-1,64,3) # (batch,64,3)对应论文中reshape操作变为M1x3 64x3 center1
        
        pc2_feat = F.relu(self.fc2_1(x_2)) #(batch,8192),8192=64*128对应论文中对FC2的M1·M2
        pc2_feat = pc2_feat.reshape(-1,128,64) # (batch,128,64),对应论文中reshape为M2xM1
        pc2_xyz =self.conv2_1(pc2_feat) # (batch,6,64),对应论文中Conv为(3M2/M1)xM1 6x64 center2
        
        pc3_feat = F.relu(self.fc1_1(x_1)) # (batch,65536),65536对应论文中对FC1的M·M2
        pc3_feat = pc3_feat.reshape(-1,512,128) # (batch,512,128),对应论文中reshape为MxM2
        pc3_feat = F.relu(self.conv1_1(pc3_feat)) # (batch,512,128)
        pc3_feat = F.relu(self.conv1_2(pc3_feat)) # (batch,256,128)
        pc3_xyz = self.conv1_3(pc3_feat) # (batch,24,128)对应论文中Conv出的(3M/M2)xM2 示例中crop_point_num设为1024,若与M同样设为512则为:12x128 fine
        
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2) # (batch,M1,1,3)对应论文中对FC3的expand为M1x1x3
        pc2_xyz = pc2_xyz.transpose(1,2) # (batch,64,6)
        pc2_xyz = pc2_xyz.reshape(-1,64,2,3) # (batch,64,2,3)
        pc2_xyz = pc1_xyz_expand+pc2_xyz # (batch,64,2,3)对应论文中FC3和FC2的结果的Add
        pc2_xyz = pc2_xyz.reshape(-1,128,3) # (batch,128,3)对应论文中Add结果的reshape为M2x3
        
        pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2) # (batch,128,1,3)对应论文中对M2x3的expand为M2x1x3
        pc3_xyz = pc3_xyz.transpose(1,2) # (batch,128,24)
        pc3_xyz = pc3_xyz.reshape(-1,128,int(self.crop_point_num/128),3) # (batch,128,8,3)对应论文中对FC1的Conv出的(3M/M2)xM2 reshape为M2x(M/M2)x3
        pc3_xyz = pc2_xyz_expand+pc3_xyz # (batch,128,8,3)对应论文中FC2和FC1的结果的Add
        pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3) # (batch,1024,3)对应论文中对上述结果的reshape为Mx3
        
        return pc1_xyz,pc2_xyz,pc3_xyz #center1 ,center2 ,fine
    
# 类似U-Net的生成器
class _unetG(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
        super(_unetG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        # 设M1=64,M2=128,M=crop_point_num
        # 线性层
        self.fc1=nn.Linear(1920,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,256)

        # 同级的线性层
        self.fc1_1=nn.Linear(1024,512*128)
        self.fc2_1=nn.Linear(512,64*128)
        self.fc3_1=nn.Linear(256,3*64)
        self.fc4_1=nn.Linear(256,3*64)

        # 各级的卷积层
        self.conv1_1=nn.Conv1d(512,512,1)
        self.conv1_2=nn.Conv1d(512,256,1)
        self.conv1_3=nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1=nn.Conv1d(128,128,1)
        self.conv2_2=nn.Conv1d(128,3*128,1)
        self.conv2_3=nn.Conv1d(384,18,1) # 9*(M2/M1)=18

        # 最后的线性层
        self.fl1=nn.Linear(1024,512)
        self.fl2=nn.Linear(512,256)
        
    def forward(self,x):
        x=self.latentfeature(x) # 1920
        x_1=F.relu(self.fc1(x)) # 1024
        x_2=F.relu(self.fc2(x_1)) # 512
        x_3=F.relu(self.fc3(x_2)) # 256
        x_4=F.relu(self.fc4(x_3)) #256

        # FC4的生成过程
        pc4_feat=self.fc4_1(x_4)
        pc4_xyz=pc4_feat.reshape(-1,64,3)
        pc4_xyz=pc4_feat.reshape(-1,64,1,3)

        # FC3的生成过程
        pc3_feat=self.fc3_1(x_3)
        pc3_xyz=pc3_feat.reshape(-1,64,1,3)

        # FC2的生成过程
        pc2_feat=self.fc2_1(x_2)
        pc2_feat=pc2_feat.reshape(-1,128,64)
        pc2_feat=F.relu(self.conv2_1(pc2_feat))
        pc2_feat=F.relu(self.conv2_2(pc2_feat))
        pc2_xyz=self.conv2_3(pc2_feat) # (batch,9,M1=64)

        # FC1的生成过程
        pc1_feat=self.fc1_1(x_1)
        pc1_feat=pc1_feat.reshape(-1,512,128)
        pc1_feat=F.relu(self.conv1_1(pc1_feat))
        pc1_feat=F.relu(self.conv1_2(pc1_feat))
        pc1_xyz=self.conv1_3(pc1_feat) # (batch,3M/M2,128)

        # 最下层的级联过程
        pc3_xyz=torch.cat((pc3_xyz,pc4_xyz),dim=2) # (batch,M1,2,3)
        pc3_xyz=pc3_xyz.reshape(-1,128,3)

        # 中间层的级联过程
        pc3_xyz_expand=torch.unsqueeze(pc3_xyz,2) # (batch,M2,1,3)
        pc2_xyz=pc2_xyz.transpose(1,2)
        pc2_xyz=pc2_xyz.reshape(-1,128,3,3)
        pc2_xyz=torch.cat((pc2_xyz,pc3_xyz_expand),dim=2) # (batch,128,4,3)
        pc2_xyz=pc2_xyz.reshape(-1,512,3)

        # 最上层的级联过程
        pc2_xyz_expand=torch.unsqueeze(pc2_xyz,2) # (batch,M=512,1,3)
        pc1_xyz=pc1_xyz.transpose(1,2)
        pc1_xyz=pc1_xyz.reshape(-1,128,int(self.crop_point_num/128),3)
        pc1_xyz=pc1_xyz.reshape(-1,512,3)
        pc1_xyz=pc1_xyz.reshape(-1,512,1,3)
        pc1_xyz=torch.cat((pc1_xyz,pc2_xyz_expand),dim=2) # (batch,512,2,3)
        pc1_xyz=pc1_xyz.reshape(-1,1024,3)

        # 对返回值点进行一次线性操作满足鉴别器输入
        pc2_xyz=pc2_xyz.view(-1,512)
        pc2_xyz=self.fl2(pc2_xyz)
        pc2_xyz=pc2_xyz.reshape(-1,256,3)
        pc1_xyz=pc1_xyz.view(-1,1024)
        pc1_xyz=self.fl1(pc1_xyz)
        pc1_xyz=pc1_xyz.reshape(-1,512,3)

        return pc3_xyz,pc2_xyz,pc1_xyz

# 鉴别点云的网络
class _netlocalD(nn.Module):
    def __init__(self,crop_point_num):
        super(_netlocalD,self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_64 = torch.squeeze(self.maxpool(x_64))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        Layers = [x_256,x_128,x_64]
        x = torch.cat(Layers,1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x
    
# 鉴别轮廓图像的网络
class _netoutlineD(nn.Module):
    def __init__(self):
        super(_netoutlineD, self).__init__()
        # 卷积层，逐步提取特征
        self.conv1 = torch.nn.Conv2d(1, 64, (3, 3), stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, (3, 3), stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 512, (3, 3), stride=2, padding=1)
        # # 池化层
        # self.maxpool = torch.nn.MaxPool2d(2, 2)
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        # 全局平均池化层
        self.gap = nn.AdaptiveAvgPool2d(1)  # 将每个通道的特征图降到1x1
        # 全连接层
        self.fc1 = nn.Linear(512, 1024)  # 根据下采样后的图像大小调整
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)
        # 批归一化层（全连接层）
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # 卷积层 + 批归一化 + 激活函数
        x = F.relu(self.bn1(self.conv1(x)))  # 输入通道为1，输出64通道
        x = F.relu(self.bn2(self.conv2(x)))  # 输入64通道，输出128通道
        x = F.relu(self.bn3(self.conv3(x)))  # 输入128通道，输出256通道
        x = F.relu(self.bn4(self.conv4(x)))  # 输入256通道，输出512通道
        # 全局平均池化 (GAP)
        x = self.gap(x)  # 将每个通道降为1x1的特征图
        # 展平操作
        x = torch.flatten(x, start_dim=1)  # 将输出展平为 512 个特征
        # 全连接层 + 批归一化 + 激活函数
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.relu(self.bn_fc3(self.fc3(x)))
        # 输出层
        x = self.fc4(x)
        return x

# 使用U-Net提取投影轮廓图像的[batch_size,128]维特征
class _UNet_Feature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_UNet_Feature, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # stride=1 保持尺寸
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # stride=1 保持尺寸
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # stride=1 保持尺寸
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # stride=1 保持尺寸
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # stride=1 保持尺寸
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # stride=1 保持尺寸
            nn.ReLU(inplace=True)
        )
        # 使用全局池化将特征图压缩成固定大小的特征向量
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc_x2 = nn.Linear(128, out_channels)  # 输出中间层的128维特征
        self.fc_x3 = nn.Linear(64, out_channels)  # 输出最终的128维特征

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        # x3 = self.decoder(x2)  # 使用解码器进行输出
        # 全局平均池化
        x2=self.global_pool(x2)
        x2=torch.flatten(x2,1)
        # 通过全连接层将特征压缩为 [batch_size, 128]
        x2 = self.fc_x2(x2)
        return x2 # 只返回x2表示图像的更高级特征
    
# 使用U-net的特征提取的图像特征与点云通过CMLP的特征结合, 利用此特征进行点云生成
class _MultiFeatureCompleteG(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num, pc_weight, img_weight):
        super(_MultiFeatureCompleteG,self).__init__()
        self.crop_point_num = crop_point_num
        self.pc_weight=pc_weight
        self.img_weight=img_weight
        self.contour_feature=_UNet_Feature(1, 128) # 输入的图像是单通道, 输出128维的特征
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        # 设M1=64,M2=128,M=crop_point_num
        # 线性层
        self.fc1=nn.Linear(2048,1024) # 由于拼接后的特征维度维128, 故修改为1920+128=2048
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,256)

        # 同级的线性层
        self.fc1_1=nn.Linear(1024,512*128)
        self.fc2_1=nn.Linear(512,64*128)
        self.fc3_1=nn.Linear(256,3*64)
        self.fc4_1=nn.Linear(256,3*64)

        # 各级的卷积层
        self.conv1_1=nn.Conv1d(512,512,1)
        self.conv1_2=nn.Conv1d(512,256,1)
        self.conv1_3=nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1=nn.Conv1d(128,128,1)
        self.conv2_2=nn.Conv1d(128,3*128,1)
        self.conv2_3=nn.Conv1d(384,18,1) # 9*(M2/M1)=18

        # 最后的线性层
        self.fl1=nn.Linear(1024,512)
        self.fl2=nn.Linear(512,256)
    
    # 前向传播(输入为3个分别为[batch_size, 2048/1024/512, 3]的点云组和[batch_size, 1, 256, 256]的投影轮廓图像)
    def forward(self, x, y):
        x=self.latentfeature(x) # 提取[batch_size,1920]维的点云特征
        y=self.contour_feature(y) # 提取的[batch_size,128]维投影轮廓图像特征
        joint_feature=torch.cat((self.pc_weight*x, self.img_weight*y), dim=1) # 在维度1上进行特征拼接

        x_1=F.relu(self.fc1(joint_feature)) # 1024, 与先前不同, 放入的是联合特征
        x_2=F.relu(self.fc2(x_1)) # 512
        x_3=F.relu(self.fc3(x_2)) # 256
        x_4=F.relu(self.fc4(x_3)) #256

        # FC4的生成过程
        pc4_feat=self.fc4_1(x_4)
        pc4_xyz=pc4_feat.reshape(-1,64,3)
        pc4_xyz=pc4_feat.reshape(-1,64,1,3)

        # FC3的生成过程
        pc3_feat=self.fc3_1(x_3)
        pc3_xyz=pc3_feat.reshape(-1,64,1,3)

        # FC2的生成过程
        pc2_feat=self.fc2_1(x_2)
        pc2_feat=pc2_feat.reshape(-1,128,64)
        pc2_feat=F.relu(self.conv2_1(pc2_feat))
        pc2_feat=F.relu(self.conv2_2(pc2_feat))
        pc2_xyz=self.conv2_3(pc2_feat) # (batch,9,M1=64)

        # FC1的生成过程
        pc1_feat=self.fc1_1(x_1)
        pc1_feat=pc1_feat.reshape(-1,512,128)
        pc1_feat=F.relu(self.conv1_1(pc1_feat))
        pc1_feat=F.relu(self.conv1_2(pc1_feat))
        pc1_xyz=self.conv1_3(pc1_feat) # (batch,3M/M2,128)

        # 最下层的级联过程
        pc3_xyz=torch.cat((pc3_xyz,pc4_xyz),dim=2) # (batch,M1,2,3)
        pc3_xyz=pc3_xyz.reshape(-1,128,3)

        # 中间层的级联过程
        pc3_xyz_expand=torch.unsqueeze(pc3_xyz,2) # (batch,M2,1,3)
        pc2_xyz=pc2_xyz.transpose(1,2)
        pc2_xyz=pc2_xyz.reshape(-1,128,3,3)
        pc2_xyz=torch.cat((pc2_xyz,pc3_xyz_expand),dim=2) # (batch,128,4,3)
        pc2_xyz=pc2_xyz.reshape(-1,512,3)

        # 最上层的级联过程
        pc2_xyz_expand=torch.unsqueeze(pc2_xyz,2) # (batch,M=512,1,3)
        pc1_xyz=pc1_xyz.transpose(1,2)
        pc1_xyz=pc1_xyz.reshape(-1,128,int(self.crop_point_num/128),3)
        pc1_xyz=pc1_xyz.reshape(-1,512,3)
        pc1_xyz=pc1_xyz.reshape(-1,512,1,3)
        pc1_xyz=torch.cat((pc1_xyz,pc2_xyz_expand),dim=2) # (batch,512,2,3)
        pc1_xyz=pc1_xyz.reshape(-1,1024,3)

        # 对返回值点进行一次线性操作满足鉴别器输入
        pc2_xyz=pc2_xyz.view(-1,512)
        pc2_xyz=self.fl2(pc2_xyz)
        pc2_xyz=pc2_xyz.reshape(-1,256,3)
        pc1_xyz=pc1_xyz.view(-1,1024)
        pc1_xyz=self.fl1(pc1_xyz)
        pc1_xyz=pc1_xyz.reshape(-1,512,3)

        return pc3_xyz,pc2_xyz,pc1_xyz

if __name__=='__main__':
    # input1 = torch.randn(64,2048,3)
    # input2 = torch.randn(64,512,3)
    # input3 = torch.randn(64,256,3)
    # input_ = [input1,input2,input3] # 随机生成的点云
    img=torch.randn(4,1,512,512) # 随机生成的图像(只有单通道)
    
    # 实例化图像鉴别器网络
    net = _netoutlineD()
    output = net(img)
    print(output)

    # # 实例化多模态点云补全网络
    # mul_net=_MultiFeatureCompleteG(3,1,[2048,512,256],512)
    # output1, output2, output3=mul_net(input_,img)
    # print(output1.size())
    # print(output2.size())
    # print(output3.size())

    # # 实例化unet
    # unet_feature=_UNet_Feature(1,128)
    # img_feature=unet_feature(img) # torch.Size([24, 128])

    # # 拓展一个通道维度
    # input_test=torch.unsqueeze(input2,1)
    # xoz,yoz,xoy=extract_outline(input_test)
    # outlinenetD=_netoutlineD(512)
    # output_outline = outlinenetD(xoz)
    # netG=_netG(3,1,[2048,512,256],512)
    # output = netG(input_)
    # unetG=_unetG(3,1,[2048,512,256],512)
    # output_unet=unetG(input_)
    # print(output[0].size(),output[1].size(),output[2].size())
    # print(output_unet[0].size(),output_unet[1].size(),output_unet[2].size())
