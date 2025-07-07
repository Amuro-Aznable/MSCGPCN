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
from utils import random_cube
from utils import divide_occlude
from utils import divide_dis
from utils import contour_projection
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default='./dataset/sunflower/', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--pc_weight', default=0.9, help='weight of point cloud input while feature fusion')
parser.add_argument('--img_weight', default=0.1, help='weight of image input while feature fusion')
parser.add_argument('--class_choice', default = '6_leaf', help = 'category of complete task')
parser.add_argument('--niter', type=int, default=231, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='./Model/mul/', help="path to load multi-view model")
parser.add_argument('--unetG', default='', help="path to load multi-view model")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outlinenetD',default='',help="path to outlinenetD(to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--if_show', default = False, help='False means do not save the contour iamges into path')
parser.add_argument('--loss_txt', default='./loss/Train_bank.txt', help=".txt path of loss details during training")
parser.add_argument('--model_path', default='./Model/mul/', help="path to save or load(when showing) model parameters")
parser.add_argument('--bank',default='./test_example/Train_bank/bank/',help="path to save projection contour in same view of each data in batch")
parser.add_argument('--axis_projection',default='./test_example/Train_bank/mul/',help="path to save multi-view pictures of the latest epoch")
parser.add_argument('--test_example',default='./test_example/Train_bank/', help="path to show the completion results")
opt = parser.parse_args()
print(opt)


def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

def camera_pos_to_str(pos):
    pos = [int(p.item()) for p in pos]
    return '{}_{}_{}'.format(pos[0], pos[1], pos[2])

if __name__=='__main__':
    test_dset = shapenet_part_loader.PartDataset( root=opt.dataroot,classification=True, class_choice=opt.class_choice, npoints=opt.pnum, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                            shuffle=False,num_workers = int(opt.workers))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    camera_position = torch.tensor([-2, 2, 2])
    # camera_position=random_cube()

    point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
    # point_netG = _unetG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
    point_netG = torch.nn.DataParallel(point_netG)
    point_netG.to(device)
    cam_str = camera_pos_to_str(camera_position)
    point_netG.load_state_dict(torch.load(opt.model_path+cam_str+'.pth', map_location=lambda storage, location: storage)['state_dict'])
    # point_netG.load_state_dict(torch.load(opt.unetG,map_location=lambda storage, location: storage)['state_dict'])  
    point_netG.eval()

    criterion_PointLoss = PointLoss().to(device)
    errG_min = 100

    print("Camera Position:{}".format(camera_position))
    
    # choice = torch.randint(0, 2, (1,))
    choice = 0
    for i, data in enumerate(test_dataloader, 0):
            
        real_point, target = data
        batch_size = real_point.size()[0]

        if choice == 0:
            input_cropped1, real_center = divide_occlude(real_point, camera_position)
        else:
            input_cropped1, real_center = divide_dis(real_point, camera_position)

        real_point = torch.unsqueeze(real_point, 1)
        # real_center = torch.unsqueeze(real_center, 1)
        input_cropped1 = torch.unsqueeze(input_cropped1,1)

        real_point = real_point.to(device)
        real_center = real_center.to(device)
        input_cropped1 = input_cropped1.to(device)

        input_cropped1 = torch.squeeze(input_cropped1,1)
        input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
        input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
        input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
        input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
        input_cropped2 = input_cropped2.to(device)
        input_cropped3 = input_cropped3.to(device)      
        input_cropped  = [input_cropped1,input_cropped2,input_cropped3]

        fake_center1,fake_center2,fake=point_netG(input_cropped)

        errG = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))#+0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))
        errG = errG.cpu()
        n = 0
        if errG.detach().numpy()>errG_min:
            pass
        else:
            errG_min = errG.detach().numpy()
            print(errG_min)
            fake =fake.cpu()
            np_fake = fake[0].detach().numpy()  # 512
            real_center = real_center.cpu()
            np_real = real_center.data[0].detach().numpy() # 512
            input_cropped1 = input_cropped1.cpu()
            np_inco = input_cropped1[0].detach().numpy() # 2048
            input_cropped2 = input_cropped2.cpu()
            input_cropped3 = input_cropped3.cpu()
            np_inco2= input_cropped2[0].detach().numpy() # 1024
            np_inco3 =input_cropped3[0].detach().numpy() # 512
            np_crop = []
            n=n+1
            k = 0
            for m in range(opt.pnum):
                if distance_squre1(np_inco[m],[0,0,0])==0.00000 and k<opt.crop_point_num:
                    k += 1
                    pass
                else:
                    np_crop.append(np_inco[m])
            # np.savetxt(opt.test_example+'/crop'+str(n)+'.csv', np_crop, fmt = "%f %f %f")
            # np.savetxt(opt.test_example+'/fake'+str(n)+'.csv', np_fake, fmt = "%f %f %f")
            # np.savetxt(opt.test_example+'/real'+str(n)+'.csv', np_real, fmt = "%f %f %f")
            np.savetxt(opt.test_example+opt.class_choice+'_'+cam_str+'_crop'+str(n)+'.txt', np_crop, fmt = "%f %f %f")
            np.savetxt(opt.test_example+opt.class_choice+'_'+cam_str+'_fake'+str(n)+'.txt', np_fake, fmt = "%f %f %f")
            np.savetxt(opt.test_example+opt.class_choice+'_'+cam_str+'_real'+str(n)+'.txt', np_real, fmt = "%f %f %f")    
