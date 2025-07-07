"""
    3.25更新:
        (1) 采用无约束的新的训练方法;
        (2) 将每个batch的点云在相同视角下生成的投影轮廓图放入"投影图像库"中, 预测点云fake所得的投影图像与图像库进行MSE损失(同一类别结构具有一定相似性);
        (3) 针对单个对象, 对fake和real采用多视角生成投影图像, 对每一视角的投影图像计算损失;
        (4) 针对投影图像的"空洞像素"问题, 使用"密度感知的半径估计"决定不同视角下投影点的像素点半径, 关键是如何判别每个点是否属于前景点;
    6.7更新:
        camera_position=random_cube(), 每个视角训练一个对应的模型.pth, 手动修改模型名字与视角相同
    6.24更新:
        增加了tensorboard可视化损失函数或其他参数下降趋势(但是需额外激活tensorboard环境(现有环境有依赖库冲突))
        运行 tensorboard --logdir='日志文件目录(注意不是指向文件, 而是所在文件夹)' --port 6006
        然后在浏览器打开http://localhost:6006/
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
from torch.utils.tensorboard import SummaryWriter
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
from utils import random_cube
from utils import divide_occlude
from utils import divide_dis
from utils import contour_projection

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
# 训练参数
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
# 模型选择与加载
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG in paper PF-Net(to continue training)")
parser.add_argument('--unetG', default='', help="path to unetG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outlinenetD',default='',help="path to outlinenetD(to continue training)")
# 策略参数和损失函数参数
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
# 保存文件
parser.add_argument('--writer_dir', default='./runs/visual_result', help = 'path to visual loss with tensorboard')
parser.add_argument('--if_show', default = False, help='False means do not save the contour iamges into path')
parser.add_argument('--loss_txt', default='./loss/Train_bank.txt', help=".txt path of loss details during training")
parser.add_argument('--model_path', default='./Model/Train_bank/', help="path to save model parameters")
parser.add_argument('--bank',default='./test_example/Train_bank/bank/',help="path to save projection contour in same view of each data in batch")
# 汇总参数
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 实例化生成器和两个鉴别器
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
# point_netG = _unetG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
point_netD = _netlocalD(opt.crop_point_num)
outline_netD=_netoutlineD()

cudnn.benchmark = True
resume_epoch=0

# 预处理图像文件的转换操作
transform_img=transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小为 [512, 512]
    transforms.ToTensor(),
])

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 

if __name__=='__main__':
    # 加载模型, 指定随机数种子
    if USE_CUDA:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        point_netG = torch.nn.DataParallel(point_netG)
        point_netD = torch.nn.DataParallel(point_netD)
        outline_netD = torch.nn.DataParallel(outline_netD)
        point_netG.to(device) 
        point_netG.apply(weights_init_normal)
        point_netD.to(device)
        point_netD.apply(weights_init_normal)
        outline_netD.to(device)
        outline_netD.apply(weights_init_normal)
    if opt.netG != '' :
        point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netG)['epoch']
    # if opt.unetG != '' :
    #     point_netG.load_state_dict(torch.load(opt.unetG,map_location=lambda storage, location: storage)['state_dict'])
    #     resume_epoch = torch.load(opt.unetG)['epoch']
    if opt.netD != '' :
        point_netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netD)['epoch']
    if opt.outlinenetD != '' :
        outline_netD.load_state_dict(torch.load(opt.outlinenetD,map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.outlinenetD)['epoch']

    print(point_netG)
    print(point_netD)
    print(outline_netD)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    
    # 加载训练集和测试集
    dset = shapenet_part_loader.PartDataset( root=opt.dataroot,classification=True, class_choice=None, npoints=opt.pnum, split='train')
    assert dset
    dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                            shuffle=True,num_workers = int(opt.workers))
    test_dset = shapenet_part_loader.PartDataset( root=opt.dataroot,classification=True, class_choice=None, npoints=opt.pnum, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                            shuffle=True,num_workers = int(opt.workers))
    
    # 判别器
    # torch.nn.BCEWithLogitsLoss用于二分类任务
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # PointLoss计算chamfer distance
    criterion_PointLoss = PointLoss().to(device)
    # 优化器
    optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
    optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
    optimizeroutD=torch.optim.Adam(outline_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)
    scheduleroutD=torch.optim.lr_scheduler.StepLR(optimizeroutD, step_size=40, gamma=0.2)

    # 输入量的初始化
    crop_point_num = int(opt.crop_point_num)
    num_batch = len(dset) / opt.batchSize
    
    # tensorboard的可视化结果
    writer = SummaryWriter(log_dir = opt.writer_dir)
    
    # 输入dummy input 在tensorboard中传入模型结构(假设只输入一维)
    dummy_input1 = torch.randn(1, 2048, 3).to(device)
    dummy_input2 = torch.randn(1, 1024, 3).to(device)
    dummy_input3 = torch.randn(1, 512, 3).to(device)
    dummy_input = [dummy_input1, dummy_input2, dummy_input3]
    writer.add_graph(point_netG, (dummy_input, ))

    if opt.D_choose == 1:
        # 启用异常检测
        torch.autograd.set_detect_anomaly(True)

        # 通过opt.niter指定迭代次数
        for epoch in range(resume_epoch,opt.niter):
            if epoch<30:
                alpha1 = 0.01
                alpha2 = 0.02
            elif epoch<80:
                alpha1 = 0.05
                alpha2 = 0.1
            else:
                alpha1 = 0.1
                alpha2 = 0.2
            
            # 每一轮迭代选定不同观测点, 同一轮迭代中的所有观测点为同一个
            camera_position = torch.tensor([-2, -2, -2]) # 指定观测点
            # camera_position=random_cube()
            
            # 选定数据缺失策略的随机数
            choice = torch.randint(0, 2, (1,))
            
            print("Epoch {} Current Camera Position:{}".format(epoch, camera_position))

            # 开始从数据集中加载一个batch的数据
            for i, data in enumerate(dataloader, 0):
                real_point, target = data
                # 数据数量并非刚好与设定的opt.batchSize相等, 所以batchsize需要动态变化
                batch_size = real_point.size()[0]
                # 根据相机位置camera_position分离点云的被遮挡点(存入为real)和非遮挡点(存入为crop, 包含了real的置零点)
                # input_cropped1.size()=torch.Size([4, 2048, 3]), real_center.size()=torch.Size([4, 512, 3])
                
                if choice == 0:
                    input_cropped1, real_center = divide_occlude(real_point, camera_position)
                else:
                    input_cropped1, real_center = divide_dis(real_point, camera_position)
                # 增加一维表示通道
                real_point = torch.unsqueeze(real_point, 1) # torch.Size([4, 1, 2048, 3])
                input_cropped1 = torch.unsqueeze(input_cropped1,1) # torch.Size([4, 1, 2048, 3])
                real_center = torch.unsqueeze(real_center,1) # torch.Size([4, 1, 512, 3])

                # 根据batch_size填充标签
                real_label = torch.FloatTensor(batch_size)
                fake_label = torch.FloatTensor(batch_size)
                real_label.resize_([batch_size, 1]).fill_(1)
                fake_label.resize_([batch_size, 1]).fill_(0)

                # 将变量放入GPU
                real_point = real_point.to(device)
                input_cropped1 = input_cropped1.to(device)
                real_center = real_center.to(device)
                real_label = real_label.to(device)
                fake_label = fake_label.to(device)

                ############################
                # (1) data prepare
                ###########################
                # 进行不同分辨率的IFPS降采样      
                real_center = Variable(real_center,requires_grad=True)
                real_center = torch.squeeze(real_center,1)
                # utils.farthest_point_sample(xyz, npoint,RAN = True),输入点云xyz,返回npoint个采样点的索引
                real_center_key1_idx = utils.farthest_point_sample(real_center,64,RAN = False)
                # utils.index_points(points, idx)，输入points=[B,N,C],idx=[B,S],返回索引点数据[B,S,C]
                real_center_key1 = utils.index_points(real_center,real_center_key1_idx)
                # real_center_key1的大小为[24,64,3]，即表示对筛选出的512个点即real_center进行IFPS采样出的64分辨率的点
                real_center_key1 =Variable(real_center_key1,requires_grad=True)

                real_center_key2_idx = utils.farthest_point_sample(real_center,128,RAN = True)
                real_center_key2 = utils.index_points(real_center,real_center_key2_idx)
                # real_center_key2的大小为[24,128,3]，即表示对筛选出的512个点即real_center进行IFPS采样出的128分辨率的点
                real_center_key2 =Variable(real_center_key2,requires_grad=True)
                # real_center_key1、real_center_key2在L2损失值计算使用
                input_cropped1 = torch.squeeze(input_cropped1,1)
                # opt.point_scales_list=[2048,1024,512],表示每个尺度的点的数量
                input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
                input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
                input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
                input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
                # input_cropped1、input_cropped2、input_cropped3分别表示每一个batch中大小为2048、1024、512分辨率的点云数据
                input_cropped1 = Variable(input_cropped1,requires_grad=True)
                input_cropped2 = Variable(input_cropped2,requires_grad=True)
                input_cropped3 = Variable(input_cropped3,requires_grad=True)
                # 将input_cropped2、input_cropped3放入GPU
                input_cropped2 = input_cropped2.to(device)
                input_cropped3 = input_cropped3.to(device)
                # 将3个分辨率的点云数据整合为input_cropped      
                input_cropped  = [input_cropped1,input_cropped2,input_cropped3]

                # 将网络设置为训练模式
                point_netG = point_netG.train()
                point_netD = point_netD.train()
                outline_netD = outline_netD.train()

                ############################
                # (2) Update D network
                ###########################

                # 生成器生成[batch_size,128,3]、[batch_size,256,3]、[batch_size,512,3]的预测点云
                fake_center1,fake_center2,fake  = point_netG(input_cropped)
                fake = torch.unsqueeze(fake, 1)
                real_center = torch.unsqueeze(real_center, 1)

                point_netD.zero_grad() # 先对鉴别点云的鉴别器梯度清零

                # real鉴别器输出及其与real标签的损失
                output = point_netD(real_center)
                errD_real = criterion(output,real_label)
                errD_real.backward(retain_graph=True)

                # fake鉴别器输出及其与fake标签的损失
                output = point_netD(fake)
                errD_fake = criterion(output, fake_label)
                errD_fake.backward(retain_graph=True)

                # 针对鉴别器的误差
                err_point_D = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (3) Update outline D network
                ############################

                outline_netD.zero_grad() # 对图像鉴别器梯度清零

                # 对同一batch中的real进行三个坐标平面视角下投影轮廓图像的绘制
                xoy_list = []
                xoz_list = []
                yoz_list = []
                # 同时对同一batch中的fake进行三个坐标平面视角下投影轮廓图像的绘制
                fake_xoy_list = []
                fake_xoz_list = []
                fake_yoz_list = []
                for m in range(batch_size):
                    # 绘制real
                    xoy, xoz, yoz = contour_projection(m, real_center[m][0], opt.bank, opt.if_show)
                    xoy =  (F.resize(xoy.unsqueeze(0), (512, 512)) > 0).float()
                    xoz =  (F.resize(xoz.unsqueeze(0), (512, 512)) > 0).float()
                    yoz =  (F.resize(yoz.unsqueeze(0), (512, 512)) > 0).float()
                    # 将图像张量存入列表, 每一坐标平面的列表中存储的是batch_size个大小为torch.Size([1, 1, 512, 512])的二值化图像
                    xoy_list.append(xoy.unsqueeze(0))
                    xoz_list.append(xoz.unsqueeze(0))
                    yoz_list.append(yoz.unsqueeze(0))
                    # 绘制fake
                    xoy, xoz, yoz = contour_projection(m, real_center[m][0], opt.bank + 'fake', opt.if_show)
                    xoy =  (F.resize(xoy.unsqueeze(0), (512, 512)) > 0).float()
                    xoz =  (F.resize(xoz.unsqueeze(0), (512, 512)) > 0).float()
                    yoz =  (F.resize(yoz.unsqueeze(0), (512, 512)) > 0).float()
                    # 将图像张量存入列表, 每一坐标平面的列表中存储的是batch_size个大小为torch.Size([1, 1, 512, 512])的二值化图像
                    fake_xoy_list.append(xoy.unsqueeze(0))
                    fake_xoz_list.append(xoz.unsqueeze(0))
                    fake_yoz_list.append(yoz.unsqueeze(0))
                
                # 对同一样本的real和fake三坐标平面的图像分别输出鉴别器结果
                # 一个batch的real各坐标平面的图像
                real_xoy = torch.cat(xoy_list, dim = 0)
                real_xoz = torch.cat(xoz_list, dim = 0)
                real_yoz = torch.cat(yoz_list, dim = 0)
                # 一个batch的fake各坐标平面的图像
                fake_xoy = torch.cat(fake_xoy_list, dim = 0)
                fake_xoz = torch.cat(fake_xoz_list, dim = 0)
                fake_yoz = torch.cat(fake_yoz_list, dim = 0)

                # real各平面鉴别器输出结果, 总的是其1/3, 并计算损失
                output_outline_xoy = outline_netD(real_xoy)
                output_outline_xoz = outline_netD(real_xoz)
                output_outline_yoz = outline_netD(real_yoz)
                output_outline = (1/3)*(output_outline_xoy + output_outline_xoz + output_outline_yoz)
                err_outlineD_real = criterion(output_outline, real_label)
                err_outlineD_real.backward(retain_graph=True)
                # fake各平面鉴别器输出结果, 总的是其1/3, 并计算损失
                output_outline_xoy = outline_netD(fake_xoy)
                output_outline_xoz = outline_netD(fake_xoz)
                output_outline_yoz = outline_netD(fake_yoz)
                output_outline = (1/3)*(output_outline_xoy + output_outline_xoz + output_outline_yoz)
                err_outlineD_fake = criterion(output_outline, fake_label)
                err_outlineD_fake.backward(retain_graph=True)

                # 图像鉴别器总误差
                err_contour_D = err_outlineD_real + err_outlineD_fake
                optimizeroutD.step()

                # bank(real)与每一fake的三平面投影做MSE
                xoy_mse_sum, xoz_mse_sum, yoz_mse_sum = 0.0, 0.0, 0.0
                for m in range(batch_size):
                    # fake[m] 和 real 所有图像之间的 MSE
                    for n in range(batch_size):
                        xoy_mse_sum += FF.mse_loss(fake_xoy[m], real_xoy[n], reduction = 'mean')
                        xoz_mse_sum += FF.mse_loss(fake_xoz[m], real_xoz[n], reduction = 'mean')
                        yoz_mse_sum += FF.mse_loss(fake_yoz[m], real_yoz[n], reduction = 'mean')
                xoy_mse = xoy_mse_sum/batch_size
                xoz_mse = xoz_mse_sum/batch_size
                yoz_mse = yoz_mse_sum/batch_size
                # 总的MSE
                total_mse = (1/3) * (xoy_mse + xoz_mse + yoz_mse)

                ############################
                # (4) Update G network: maximize log(D(G(z)))
                ###########################
                # 清空生成器梯度
                point_netG.zero_grad()

                # 需要重新计算output和output_outline, 因为前面已经使用这两个数据进行梯度更新
                output = point_netD(fake)
                output_outline_xoy = outline_netD(fake_xoy)
                output_outline_xoz = outline_netD(fake_xoz)
                output_outline_yoz = outline_netD(fake_yoz)
                output_outline = (1/3)*(output_outline_xoy + output_outline_xoz + output_outline_yoz)

                # fake与real的标签的损失
                errG_D = criterion(output, real_label)
                errG_contour_D = criterion(output_outline, real_label)

                # fake与real的CD_LOSS
                CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))

                # 多分辨率的CD损失
                errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
                +alpha1*criterion_PointLoss(fake_center1,real_center_key1)\
                +alpha2*criterion_PointLoss(fake_center2,real_center_key2)

                # 生成器总误差 = 多分辨率CD损失 + 点云鉴别器损失 + 图像鉴别器损失 + MSE
                # errG = opt.wtl2 * errG_l2 + 0.45 * (1 -opt.wtl2) * errG_D + 0.45 * (1 -opt.wtl2) * errG_contour_D + 0.1 * (1 -opt.wtl2) * total_mse
                # 不用MSE
                errG = opt.wtl2 * errG_l2 + 0.5 * (1 -opt.wtl2) * errG_D + 0.5 * (1 -opt.wtl2) * errG_contour_D
                errG.backward()
                optimizerG.step()

                print('Epoch:[{}/{}], Batch:[{}/{}]:\nLoss_Point: {:.4f}/ Loss_Contour: {:.4f}/ MSE: {:.4f}/ Loss_G: {:.4f}/ Loss_G_Contour: {:.4f}/ Loss_L2: {:.4f}/ Loss_total: {:.4f}/ CD_Loss: {:.4f}'.format(
                    epoch, opt.niter, i, len(dataloader),
                    err_point_D.data, err_contour_D, total_mse, errG_D.data, errG_contour_D, errG_l2, errG, CD_LOSS
                ))
                f = open(opt.loss_txt, 'a')
                f.write('\n' + 'Epoch:[{}/{}], Batch:[{}/{}]:\nLoss_Point: {:.4f}/ Loss_Contour: {:.4f}/ MSE: {:.4f}/ Loss_G: {:.4f}/ Loss_G_Contour: {:.4f}/ Loss_L2: {:.4f}/ Loss_total: {:.4f}/ CD_Loss: {:.4f}'.format(
                    epoch, opt.niter, i, len(dataloader),
                    err_point_D.data, err_contour_D, total_mse, errG_D.data, errG_contour_D, errG_l2, errG, CD_LOSS
                ))

                # 每10个batch做一次测试
                if i % 10 ==0 and i !=0:
                    print('After, ',i,'-th batch')
                    f.write('\n'+'After, '+str(i)+'-th batch')
                    for i, data in enumerate(test_dataloader, 0):
                        real_point, target = data
                        batch_size = real_point.size()[0]
                        input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                        # input_cropped1.size()=torch.Size([4, 2048, 3]), real_center.size()=torch.Size([4, 512, 3])
                        if choice == 0:
                            input_cropped1, real_center = divide_occlude(real_point, camera_position)
                        else:
                            input_cropped1, real_center = divide_dis(real_point, camera_position)
                        real_point = torch.unsqueeze(real_point, 1)
                        input_cropped1 = torch.unsqueeze(input_cropped1, 1)
                        real_center = torch.unsqueeze(real_center, 1)
                        real_center = real_center.to(device)
                        real_center = torch.squeeze(real_center,1)
                        input_cropped1 = input_cropped1.to(device) 
                        input_cropped1 = torch.squeeze(input_cropped1,1)
                        input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
                        input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
                        input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
                        input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
                        input_cropped1 = Variable(input_cropped1,requires_grad=False)
                        input_cropped2 = Variable(input_cropped2,requires_grad=False)
                        input_cropped3 = Variable(input_cropped3,requires_grad=False)
                        input_cropped2 = input_cropped2.to(device)
                        input_cropped3 = input_cropped3.to(device)      
                        input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
                        point_netG.eval()

                        fake_center1,fake_center2,fake  = point_netG(input_cropped)
                        CD_loss = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
                        print('test result:',CD_loss)
                        f.write('\n'+'test result:  %.4f'%(CD_loss))
                        break
                f.close()
                
            schedulerD.step()
            schedulerG.step()
            scheduleroutD.step()
            
            writer.add_scalar('Loss/errG_l2', errG_l2, epoch)
            writer.add_scalar('Loss/CD_loss', CD_loss, epoch)
            
            # 每隔10次epoch保存生成器和鉴别器的结果
            if epoch% 10 == 0:   
                torch.save({'epoch':epoch+1,
                            'state_dict':point_netG.state_dict()},
                            opt.model_path+'/point_netG'+str(epoch)+'.pth' )
                torch.save({'epoch':epoch+1,
                            'state_dict':point_netD.state_dict()},
                            opt.model_path+'/point_netD'+str(epoch)+'.pth' )
                torch.save({'epoch':epoch+1,
                            'state_dict':outline_netD.state_dict()},
                            opt.model_path+'/contour_netD'+str(epoch)+'.pth' )