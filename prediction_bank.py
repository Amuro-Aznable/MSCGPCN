"""
    读取.txt文件, 即读取不完整点云, 经过多分辨率降采样后送入网络中;
    依次选择文件夹中所有的预训练模型, 同时保存各模型的生成结果(注意: opt.netG或pot.unetG输入的是包含.pth模型的文件夹而并非Train时加载的单个.pth文件)
"""
import os
import glob
import cv2
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch
from torchvision.transforms import functional as F
import torch.nn.functional as FF
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
import data_utils as d_utils
import shapenet_part_loader
from model_PFNet import _MultiFeatureCompleteG,_netlocalD, _netoutlineD, _netG, _unetG
from utils import divide_occlude
from utils import divide_dis

# 加载参数
parser = argparse.ArgumentParser()
# 数据集和相机参数
parser.add_argument('--dataroot',  default='./dataset/sunflower/', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
# 模型参数
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--pc_weight', default=0.9, help='weight of point cloud input while feature fusion')
parser.add_argument('--img_weight', default=0.1, help='weight of image input while feature fusion')
parser.add_argument('--class_choice', default = '6_leaf', help = 'category of complete task')
# 训练参数
parser.add_argument('--niter', type=int, default=231, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
# 模型选择与加载
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='./Model/mul/', help="path to load multi-view model")
parser.add_argument('--unetG', default='', help="path to load multi-view model")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outlinenetD',default='',help="path to outlinenetD(to continue training)")
# 策略参数和损失函数参数
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
# 加载文件
parser.add_argument('--pc_file', default='D:/work/completion_outline/prediction_example/real/10_point_cloud.txt', help='file of incomplete point cloud')
# 保存文件
parser.add_argument('--if_show', default = False, help='False means do not save the contour iamges into path')
parser.add_argument('--loss_txt', default='./loss/Train_bank.txt', help=".txt path of loss details during training")
parser.add_argument('--model_path', default='./Model/mul/', help="path to save or load(when showing) model parameters")
parser.add_argument('--bank',default='./test_example/Train_bank/bank/',help="path to save projection contour in same view of each data in batch")
parser.add_argument('--test_example',default='./test_example/Train_bank/', help="path to show the completion results")
parser.add_argument('--prediction_example', default='./prediction_example/real/', help='path to save results of completion')
# 汇总参数
opt = parser.parse_args()
print(opt)

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 读取.txt文件并转为tensor
    incomplete_pc_np = np.loadtxt(opt.pc_file)
    # 提取前三列作为点云, 并归一化到[-1,1]
    points = incomplete_pc_np[:, :3]
    # 获取 x, y, z 的最小值和最大值
    min_vals = points.min(axis=0)  # 对每列求最小值
    max_vals = points.max(axis=0)  # 对每列求最大值
    # 对点云数据进行归一化到 [-1, 1], 并转为tensor
    incomplete_pc_np = 2 * (points - min_vals) / (max_vals - min_vals) - 1
    incomplete_pc = torch.from_numpy(incomplete_pc_np).float() # torch.Size([N, 3])
    incomplete_pc = incomplete_pc.unsqueeze(0)

    # 先对输入的incomplete point cloud降采样
    if incomplete_pc.size()[1] > 1536:
        incomplete_pc_idx = utils.farthest_point_sample(incomplete_pc, 1536, RAN=True)
        input_cropped1 = utils.index_points(incomplete_pc, incomplete_pc_idx)

    # 再将incomplete point cloud 拼接上512个0点作为输入
    zeros = torch.zeros(512, 3)
    zeros = zeros.unsqueeze(0)
    input_cropped1 = torch.cat([input_cropped1, zeros], dim = 1)
    # input_cropped1 = input_cropped1.unsqueeze(0)  # 在第0维增加一个 batch 维
    input_cropped1 = input_cropped1.to(device)
    # 输入的多分辨率降采样
    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
    input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
    input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
    # 将input_cropped2、input_cropped3放入GPU
    input_cropped2 = input_cropped2.to(device)
    input_cropped3 = input_cropped3.to(device)
    # 将3个分辨率的点云数据整合为input_cropped      
    input_cropped  = [input_cropped1,input_cropped2,input_cropped3]

    # 实例化生成器
    point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
    # point_netG = _unetG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
    point_netG = torch.nn.DataParallel(point_netG)
    point_netG.to(device)

    # 获取所有.pth文件路径（只选择 .pth 文件）
    model_files = [f for f in os.listdir(opt.netG) if f.endswith('.pth')]

    # 用每个模型预测incomplete_pc
    for model_file in model_files:
        # 生成模型文件路径
        model_file_path = os.path.join(opt.netG, model_file)
        point_netG.load_state_dict(torch.load(model_file_path, map_location=lambda storage, location: storage)['state_dict'])
        point_netG.eval()

        # 预测点云
        fake_center1,fake_center2,fake = point_netG(input_cropped)

        # 保存预测点云
        fake = fake.cpu()
        np_fake = fake[0].detach().numpy()
        np.savetxt(opt.prediction_example + 'fake_{}.txt'.format(str(model_file)), np_fake, fmt = "%f %f %f")
    
    # 保存输入点云便于观测
    input_cropped1_np = input_cropped1.cpu()
    input_cropped1_np = input_cropped1_np[0].detach().numpy()
    np.savetxt(opt.prediction_example + 'input.txt', input_cropped1_np, fmt = "%f %f %f")