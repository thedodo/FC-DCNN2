"""
(c) DI Dominik Hirner BSc. 
Institute for graphics and vision (ICG)
University of Technology Graz, Austria
e-mail: dominik.hirner@tugraz.at
"""

import os
import shutil
import sys
import glob
import numpy as np
from numpy import inf
import cv2
import matplotlib.pyplot as plt
import re
import numpy.matlib

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import random
import configparser


config = configparser.ConfigParser()

config.read(sys.argv[1])


#continue from one of our trained weights
transfer_train = config.getboolean('PARAM','transfer_train')

#KITTI, MB or ETH
dataset = config['PARAM']['dataset']
#used as prefix for saved weights
model_name = config['PARAM']['model_name']

#folder with training data
input_folder = config['PARAM']['input_folder']
batch_size = int(config['PARAM']['batch_size'])
nr_batches = int(config['PARAM']['nr_batches'])
nr_epochs = int(config['PARAM']['nr_epochs'])
num_conv_feature_maps = int(config['PARAM']['num_conv_feature_maps'])
save_weights = int(config['PARAM']['save_weights'])

#needs to be odd
#size of patch-crops fed into the networ
patch_size = int(config['PARAM']['patch_size'])

ps_h = int(patch_size/2)

#range for offset of o_neg
r_low = int(config['PARAM']['r_low'])
r_high = int(config['PARAM']['r_high'])


print("Transfer train: " ,transfer_train)
print("Dataset: " ,dataset)
print("Model name: " ,model_name)
print("Input folder: " ,input_folder)
print("Batch-size: " ,batch_size)
print("Number of Epochs: " ,nr_epochs)
print("Patch size: ", patch_size)
print("r_low: ", r_low)
print("r_high", r_high)
print("#Feature-maps per layer: " ,num_conv_feature_maps)
print("Save weights every epochs: " ,save_weights)



def loadMB():
    
    left_filelist = glob.glob(input_folder + '*/im0.png')
    right_filelist = glob.glob(input_folder + '*/im1.png')
    disp_filelist = glob.glob(input_folder + '*/disp0GT.pfm')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)
    
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(left_filelist)):
        
        cur_left = cv2.imread(left_filelist[i])
        cur_right = cv2.imread(right_filelist[i])
        cur_disp,_ = readPFM(disp_filelist[i])
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list



def loadETH3D():
    
    left_filelist = glob.glob(input_folder + '*/im0.png')
    right_filelist = glob.glob(input_folder + '*/im1.png')
    disp_filelist = glob.glob(input_folder + '*/disp0GT.pfm')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)
    
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(left_filelist)):
        
        cur_left = cv2.imread(left_filelist[i])
        cur_right = cv2.imread(right_filelist[i])
        cur_disp,_ = readPFM(disp_filelist[i])
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list


def loadKitti2012():

    left_filelist = glob.glob(input_folder + 'colored_0/*.png')
    right_filelist = glob.glob(input_folder + 'colored_1/*.png')
    disp_filelist = glob.glob(input_folder + 'disp_noc/*.png')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)

    left_elem_list = []
    for left_im in left_filelist:

        left_im_el = left_im.split('/')[-1]
        left_elem_list.append(left_im_el)

    left_elem_list = sorted(left_elem_list)


    right_elem_list = []
    for right_im in right_filelist:

        right_im_el = right_im.split('/')[-1]
        right_elem_list.append(right_im_el)

    right_elem_list = sorted(right_elem_list)



    gt_elem_list = []
    for gt_im in disp_filelist:

        gt_im_el = gt_im.split('/')[-1]
        gt_elem_list.append(gt_im_el)

    gt_elem_list = sorted(gt_elem_list)


    inters_list = set(left_elem_list) & set(right_elem_list) & set(gt_elem_list)
   
    inters_list = list(inters_list)
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(inters_list)):
        
        left_im = input_folder + 'colored_0/' + inters_list[i]
        right_im = input_folder + 'colored_1/' + inters_list[i]
        disp_im =  input_folder + 'disp_noc/' + inters_list[i] 
       
        cur_left = cv2.imread(left_im)
        cur_right = cv2.imread(right_im)
        cur_disp = cv2.imread(disp_im)
        
        cur_disp = np.mean(cur_disp,axis=2) 
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list


def loadKitti2015():

    left_filelist = glob.glob(input_folder + 'image_2/*.png')
    right_filelist = glob.glob(input_folder + 'image_3/*.png')
    disp_filelist = glob.glob(input_folder + 'disp_noc_0/*.png')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)

    left_elem_list = []
    for left_im in left_filelist:

        left_im_el = left_im.split('/')[-1]
        left_elem_list.append(left_im_el)

    left_elem_list = sorted(left_elem_list)


    right_elem_list = []
    for right_im in right_filelist:

        right_im_el = right_im.split('/')[-1]
        right_elem_list.append(right_im_el)

    right_elem_list = sorted(right_elem_list)



    gt_elem_list = []
    for gt_im in disp_filelist:

        gt_im_el = gt_im.split('/')[-1]
        gt_elem_list.append(gt_im_el)

    gt_elem_list = sorted(gt_elem_list)


    inters_list = set(left_elem_list) & set(right_elem_list) & set(gt_elem_list)
   
    inters_list = list(inters_list)
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(inters_list)):
        
        left_im = input_folder + 'image_2/' + inters_list[i]
        right_im = input_folder + 'image_3/' + inters_list[i]
        disp_im =  input_folder + 'disp_noc_0/' + inters_list[i] 
       
        cur_left = cv2.imread(left_im)
        cur_right = cv2.imread(right_im)
        cur_disp = cv2.imread(disp_im)
        
        cur_disp = np.mean(cur_disp,axis=2) 
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list


class SiameseBranch(nn.Module):
    def __init__(self,img_ch=1):
        super(SiameseBranch,self).__init__()
        
        self.Tanh = nn.Tanh()        
        self.Conv1 = nn.Conv2d(img_ch, num_conv_feature_maps, kernel_size = 3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv2 = nn.Conv2d(num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv3 = nn.Conv2d(2*num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv4 = nn.Conv2d(3*num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1,bias=True)
        self.Conv5 = nn.Conv2d(4*num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        
        
    def forward(self,x_in):
        
        x1 = self.Conv1(x_in) 
        x1 = self.Tanh(x1)
                
        x2 = self.Conv2(x1) 
        x2 = self.Tanh(x2)
        
        d2 = torch.cat((x1,x2),dim=1)
        
        x3 = self.Conv3(d2) 
        x3 = self.Tanh(x3)
        
        d3 = torch.cat((x1,x2,x3),dim=1)
        
        x4 = self.Conv4(d3)
        x4 = self.Tanh(x4)
        
        d4 = torch.cat((x1,x2,x3,x4),dim=1)
        
        x5 = self.Conv5(d4)
        x5 = self.Tanh(x5)
        
        return x5
    
    
branch = SiameseBranch()
branch = branch.cuda()


##python3 version!!!!
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)
        
Tensor = torch.cuda.FloatTensor
cos = torch.nn.CosineSimilarity()


def getBatch():
    
    ridx = np.random.randint(0,len(left_list),1)
    
    cur_left = left_list[ridx[0]]
    cur_right = right_list[ridx[0]]
    gt_im = gt_list[ridx[0]]
    
    #convert to grayscale
    left_im = np.mean(cur_left, axis = 2) 
    right_im = np.mean(cur_right, axis = 2) 
    
    batch_xl = np.zeros((batch_size,patch_size,patch_size))
    batch_xr_pos = np.zeros((batch_size,patch_size,patch_size))
    batch_xr_neg = np.zeros((batch_size,patch_size,patch_size))
    
    for el in range(batch_size):
        
        #get random position
        h,w = left_im.shape
        r_h = 0
        r_w = 0
        d = 0
        #Draw for random position
        while True:
            r_h = random.sample(range(ps_h,h-(ps_h+1)), 1)
            r_w = random.sample(range(ps_h,w-(ps_h+1)),1)            
            if(not np.isinf(gt_im[r_h,r_w])):
                d = int(np.round(gt_im[r_h,r_w]))
                if((r_w[0]-ps_h-d-1) >= 0):
                     if((r_w[0]+(ps_h+1)-d+1) <= w):
                        break
        
        d = int(np.round(gt_im[r_h,r_w]))
                
        cur_left = left_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), r_w[0]-ps_h:r_w[0]+(ps_h+1)]
        #choose offset
        
        o_pos = 0                
        cur_right_pos = right_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), (r_w[0]-ps_h-d+o_pos):(r_w[0]+(ps_h+1)-d+o_pos)]
        
        #get negativ patch with random offset withing [r_low,r_high], 50/50 chance if it is off to the left or right
        o_neg = 0
        while True:
            o_neg = random.sample(range(r_low,r_high), 1)
            if np.random.randint(-1, 1) == -1:
                o_neg = -o_neg[0]
            else:
                o_neg = o_neg[0]
            #try without d-+1   and(o_neg != (d-1)) and(o_neg != (d+1))
            if((o_neg != d) and ((r_w[0]-ps_h-d+o_neg) > 0)  and ((r_w[0]+(ps_h+1)-d+o_neg) < w)):
                break
        
        cur_right_neg = right_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), (r_w[0]-ps_h-d+o_neg):(r_w[0]+(ps_h+1)-d+o_neg)]

        
        batch_xl[el,:,:] = cur_left
        batch_xr_pos[el,:,:] = cur_right_pos
        batch_xr_neg[el,:,:] = cur_right_neg     
        
        
    batch_xl = np.reshape(batch_xl, [batch_size,1,patch_size,patch_size])
    batch_xr_pos = np.reshape(batch_xr_pos, [batch_size,1,patch_size,patch_size])
    batch_xr_neg = np.reshape(batch_xr_neg, [batch_size,1,patch_size,patch_size])
    
    return batch_xl, batch_xr_pos, batch_xr_neg


pytorch_total_params = sum(p.numel() for p in branch.parameters() if p.requires_grad)
print("Nr feat: " ,pytorch_total_params)

def my_hinge_loss(s_p, s_n):
    margin = 0.2
    relu = torch.nn.ReLU()
    relu = relu.cuda()
    loss = relu(-((s_p - s_n) - margin))
    return loss

if(dataset == 'KITTI2012'):
    left_list, right_list, gt_list = loadKitti2012()
if(dataset == 'KITTI2015'):
    left_list, right_list, gt_list = loadKitti2015()
if(dataset == 'MB'):
    left_list, right_list, gt_list = loadMB()    
if(dataset == 'ETH'):
    left_list, right_list, gt_list = loadETH3D()
    
    
nr_samples = len(left_list)

#KITTI, MB or ETH
if(transfer_train):
    if(dataset == 'KITTI'):
        
        print('Load weights from KITTI')
        branch.load_state_dict(torch.load('weights/kitti'))
        
    if(dataset == 'MB'):
        print('Load weights from Middlebury')
        branch.load_state_dict(torch.load('weights/mb'))
        
    if(dataset == 'ETH'):
        print('Load weights from ETH3D')
        branch.load_state_dict(torch.load('weights/eth3d'))


optimizer_G = optim.Adam(branch.parameters(), lr=0.00006)

left_patches = []
right_pos_patches = []
right_neg_patches = []

loss_list = []

for i in range(nr_epochs):
            
    epoch_loss = 0.0        
    for cur_batch in range(batch_size):
         #reset gradients
        optimizer_G.zero_grad()

        batch_xl, batch_xr_pos, batch_xr_neg = getBatch()
        bs, c, h, w = batch_xl.shape
        batch_loss = 0.0

        xl = Variable(Tensor(batch_xl.astype(np.uint8)))
        xr_pos = Variable(Tensor(batch_xr_pos.astype(np.uint8)))
        xr_neg = Variable(Tensor(batch_xr_neg.astype(np.uint8)))        

        left_out = branch(xl)
        right_pos_out = branch(xr_pos)
        right_neg_out = branch(xr_neg)
        
        sp = cos(left_out, right_pos_out)
        sn = cos(left_out, right_neg_out)            
        
        batch_loss = my_hinge_loss(sp, sn)
        batch_loss = batch_loss.mean()      

        batch_loss.backward()
        optimizer_G.step()
        epoch_loss = epoch_loss + batch_loss
    
    epoch_loss = batch_loss/nr_batches        
    if(i % save_weights == 0):
        torch.save(branch.state_dict(), './save_weights/' + model_name + '_%04i' %(i)) 
        print("EPOCH: {} loss: {}".format(i,epoch_loss))
        
        
val, idx = min((val, idx) for (idx, val) in enumerate(loss_list))

plt.figure()
plt.plot(loss_list,'k')
plt.plot(loss_list, 'r*')