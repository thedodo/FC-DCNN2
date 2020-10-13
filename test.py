"""
(c) DI Dominik Hirner BSc. 
Institute for graphics and vision (ICG)
University of Technology Graz, Austria
e-mail: dominik.hirner@tugraz.at
"""
import sys
import numpy as np
import cv2
import re
import numpy.matlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from typing import Tuple
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
import argparse

def main():
    
    parser = argparse.ArgumentParser(description='FC-DCNN disparity networ')
    
    parser.add_argument('--left', help='path to left rectified image')
    parser.add_argument('--weights', help='path to the trained weights')
    parser.add_argument('--right', help='path to right image')
    parser.add_argument('--max_disp', help='disparity search range', type=int)
    parser.add_argument('--out', help='path and/or name for output')
    args = parser.parse_args()
    
    weight = args.weights
    left_im = args.left
    right_im = args.right
    max_disp = args.max_disp
    out_fn = args.out
    
    branch.load_state_dict(torch.load(weight))
    disp_s, disp = TestImage(left_im, right_im, max_disp, out_fn, filtered = True, lr_check = True, fill = True)


num_conv_feature_maps = 64
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
    
Tensor = torch.cuda.FloatTensor
cos = torch.nn.CosineSimilarity()

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

def _compute_binary_kernel(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: Tuple[int, ...] = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.MedianBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel: torch.Tensor = _compute_binary_kernel(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input: torch.Tensor):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # map the local window to single vector
        features: torch.Tensor = F.conv2d(
            input, kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median



# functiona api
def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    See :class:`~kornia.filters.MedianBlur` for details.
    """
    return MedianBlur(kernel_size)(input)


def filterCostVolMedian(cost_vol):
    
    d,h,w = cost_vol.shape
    cost_vol = cost_vol.unsqueeze(0)
    
    for disp in range(d):
        cost_vol[:,disp,:,:] = median_blur(cost_vol[:,disp,:,:].unsqueeze(0), (5,5))
        
    return torch.squeeze(cost_vol)


def filterCostVolBilat(cost_vol,left):
    
    left = np.mean(left,axis=2)
    leftT = Variable(Tensor(left))
    leftT = leftT.unsqueeze(0).unsqueeze(0)

    d,h,w = cost_vol.shape  
    
    f = GuidedFilter(8,10).cuda() #0.001
    
    for disp in range(d):
        cur_slice =  cost_vol[disp,:,:]
        cur_slice = cur_slice.unsqueeze(0).unsqueeze(0)
        
        inputs = [leftT, cur_slice]

        test = f(*inputs)
        cost_vol[disp,:,:] = np.squeeze(test)
    return cost_vol


def createCostVol(left_im,right_im,max_disp):
    
    print('Creating cost-volume....')
    left_im = np.mean(left_im, axis=2)
    right_im = np.mean(right_im, axis=2)
            
    a_h, a_w = left_im.shape

    left_im = np.reshape(left_im, [1,1,a_h,a_w])
    right_im = np.reshape(right_im, [1,1,a_h,a_w])
    
    with torch.no_grad():

        left_imT = Variable(Tensor(left_im.astype(np.uint8)))
        right_imT = Variable(Tensor(right_im.astype(np.uint8)))

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        cost_volT = Variable(Tensor(cost_vol))   

        #0 => max_disp => one less disp!
        for disp in range(0,max_disp+1):
            if(disp == 0):
                sim_score = cos(left_feat, right_feat)
                cost_volT[disp,:,:] = torch.squeeze(sim_score) 
            else:
                right_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)                      
                right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
                right_appended = torch.cat([right_shift,right_feat],3)

                _,f,h_ap,w_ap = right_appended.shape
                right_shifted[:,:,:,:] = right_appended[:,:,:,:(w_ap-disp)]
                sim_score = cos(left_feat, right_shifted)
                cost_volT[disp,:,:] = torch.squeeze(sim_score)              
                
    print('Done')            
    return cost_volT


def createCostVolRL(left_im,right_im,max_disp):

    print('Create cost-volume right-to-left...')
    left_im = np.mean(left_im, axis=2)
    right_im = np.mean(right_im, axis=2)

    a_h, a_w = left_im.shape

    left_im = np.reshape(left_im, [1,1,a_h,a_w])
    right_im = np.reshape(right_im, [1,1,a_h,a_w])

    with torch.no_grad():
        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)

        _,f,h,w = left_feat.shape
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        
        cost_volT = Variable(Tensor(cost_vol))

        for disp in range(0,max_disp+1):
            if(disp == 0):
                sim_score = cos(right_feat, left_feat)
                cost_volT[disp,:,:] = torch.squeeze(sim_score)
            else:
                left_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)
                left_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)
                left_appended = torch.cat([left_feat,left_shift],3)
                
                _,f,h_ap,w_ap = left_appended.shape
                left_shifted[:,:,:,:] = left_appended[:,:,:,disp:w_ap]
                sim_score = cos(right_feat, left_shifted)                
                cost_volT[disp,:,:] = torch.squeeze(sim_score)
                
    print('Done')           
    return cost_volT

def LR_Check(first_output, second_output):  
    
    print('Check for inconsistencies...')
    h,w = first_output.shape
        
    line = np.array(range(0, w))
    idx_arr = np.matlib.repmat(line,h,1)
    
    dif = idx_arr - first_output
    
    first_output[np.where(dif <= 0)] = 0
    
    first_output = first_output.astype(np.int)
    second_output = second_output.astype(np.int)
    dif = dif.astype(np.int)
    
    second_arr_reordered = np.array(list(map(lambda x, y: y[x], dif, second_output)))
    
    dif_LR = np.abs(second_arr_reordered - first_output)
    first_output[np.where(dif_LR >= 1.1)] = 0
    
    first_output = first_output.astype(np.float32)
    first_output[np.where(first_output == 0.0)] = np.nan
    print('Done')       
    return first_output


def FillIncons(mask, disp):

    #limit for consistent point search
    max_search = 30
    w = mask.shape[1]
    h = mask.shape[0] 
    
    #BG
    idc = np.argwhere(np.isnan(disp))    
    for curnan in range(len(idc)):
        curnanh = idc[curnan][0]
        curnanw = idc[curnan][1]        
        if(mask[curnanh,curnanw] == 0):
            
            #whole scanline is nan => disp is 0
            if(all(np.isnan(disp[curnanh,:]))):
                #hole line set to 0!
                disp[curnanh,:] = 0.0
            #all px to the left are NaN
            if(all(np.isnan(disp[curnanh,0:curnanw]))):
                #go to the right
                curw = curnanw
                fill = 0
                while(np.isnan(disp[curnanh,curw]) and mask[curnanh,curnanw] == 0):
                    curw = curw +1
                    fill = disp[curnanh,curw]
                disp[curnanh,curnanw] = fill  
            #else go left
            else:
                curw = curnanw
                fill = 0
                while(np.isnan(disp[curnanh,curw]) and mask[curnanh,curnanw] == 0):
                    curw = curw -1
                    fill = disp[curnanh,curw]
                disp[curnanh,curnanw] = fill 
    #FG
    idcFG = np.argwhere(np.isnan(disp))
    for curnan in range(len(idcFG)):
        
        curnanh = idcFG[curnan][0]
        curnanw = idcFG[curnan][1]
      
        left = 0
        right = 0
        above = 0
        under = 0

        r_above = 0
        l_above = 0
        r_under = 0
        l_under = 0      
        
        if(curnanw == 0):
            left = 0
        else:
            left = int(disp[curnanh,curnanw-1])
        counter = 0                                    
        while(np.isnan(disp[curnanh,curnanw+counter])):
            counter = counter +1                       
            if((curnanw+counter) >= w or counter >= max_search):
                right = 0
                break
            right = disp[curnanh,curnanw+counter]
        counter = 0                                    
        while(np.isnan(disp[curnanh+counter,curnanw])):
            counter = counter +1                       
            if((curnanh+counter) >= h or counter >= max_search):
                above = 0
                break       
            above = disp[curnanh+counter,curnanw]
        if(curnanh == 0):
            under = 0
        else:
            under = disp[curnanh-1,curnanw]
        
        counter = 0                                    
        while(np.isnan(disp[curnanh+counter,curnanw+counter])):
            counter = counter +1
            if((curnanh+counter) >= h or counter >= max_search):
                r_above = 0
                break
            if((curnanw+counter) >= w):
                r_above = 0
                break                        
            r_above = disp[curnanh+counter,curnanw+counter]   
        
        if(curnanh == 0 or curnanw == 0):
            l_under = 0
        else:
            l_under = disp[curnanh-1,curnanw-1]  
        
        counter = 0      
        while(np.isnan(disp[curnanh+counter,curnanw-counter])):
            counter = counter +1
            if((curnanh+counter) >= h):
                l_above = 0
                break
            if((curnanw-counter) <= 0 or counter >= max_search):
                l_above = 0
                break
            l_above = disp[curnanh+counter,curnanw-counter]

        if(curnanh == 0 or curnanw >= w-1):
            r_under = 0
        else:
            r_under = disp[curnanh-1,curnanw+1]
        
        fill = np.median([left,right,above,under,r_above,l_above,r_under,l_under])
        disp[curnanh,curnanw] = fill
    return disp


def TestImage(fn_left, fn_right, max_disp, im_to_save, filtered = True, lr_check = True, fill = True):
    
    left = cv2.imread(fn_left)
    right = cv2.imread(fn_right)
    disp_map = []
    
    if(filtered):
        
        cost_vol = createCostVol(left,right,max_disp)
        cost_vol = filterCostVolMedian(cost_vol) 
        cost_vol = filterCostVolMedian(cost_vol) 
        cost_vol = filterCostVolMedian(cost_vol)
        cost_vol = filterCostVolMedian(cost_vol)
        
        cost_vol_filteredn = filterCostVolBilat(cost_vol,left)
        cost_vol_filteredn = np.squeeze(cost_vol_filteredn.cpu().data.numpy())        
        
        disp = np.argmax(cost_vol_filteredn, axis=0) 
        writePFM(im_to_save + '.pfm', disp.astype(np.float32), scale=1)                
        if(lr_check):
            cost_vol_RL = createCostVolRL(left,right,max_disp)
            cost_vol_RL = filterCostVolMedian(cost_vol_RL)
            cost_vol_RL = filterCostVolMedian(cost_vol_RL)   
            cost_vol_RL = filterCostVolMedian(cost_vol_RL)
            cost_vol_RL = filterCostVolMedian(cost_vol_RL)
            
            cost_vol_RL_fn = filterCostVolBilat(cost_vol_RL,right)
            cost_vol_RL_fn = np.squeeze(cost_vol_RL_fn.cpu().data.numpy())
            
            disp_map_RL = np.argmax(cost_vol_RL_fn, axis=0)  
            disp_map = LR_Check(disp.astype(np.float32), disp_map_RL.astype(np.float32))
            writePFM(im_to_save + '_s.pfm', disp_map.astype(np.float32), scale=1)      
        
    else:
        
        cost_vol = createCostVol(left,right,max_disp)
        cost_vol = np.squeeze(cost_vol.cpu().data.numpy())
        
        disp = np.argmax(cost_vol, axis=0)        
        writePFM(im_to_save + '.pfm', disp.astype(np.float32), scale=1)      
        
        if(lr_check):
            
            cost_vol_RL = createCostVolRL(left,right,max_disp)
            cost_vol_RL = np.squeeze(cost_vol_RL.cpu().data.numpy())
            disp_map_RL = np.argmax(cost_vol_RL, axis=0)       
            disp_map = LR_Check(disp.astype(np.float32), disp_map_RL.astype(np.float32))
            writePFM(im_to_save + '_s.pfm', disp_map.astype(np.float32), scale=1)           
            
    if(fill):
    
        disp = np.array(disp_map)
        im_disp = Image.fromarray(disp) 
        im_disp = np.dstack((im_disp, im_disp, im_disp)).astype(np.uint8)    

        h,w = disp.shape

        shifted = cv2.pyrMeanShiftFiltering(im_disp, 7, 7)

        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 1,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        kernel = np.ones((5,5), np.uint8)

        dilation = cv2.dilate(thresh,kernel,iterations = 3)
        mask = cv2.erode(dilation, kernel, iterations=2)    

        cv2.imwrite(im_to_save + 'bilat_and_med_mask.png',mask * 255)

        disp_filled = FillIncons(mask, disp)
        writePFM(im_to_save + '_filled.pfm',disp_filled)
    return disp_map, disp


if __name__ == "__main__":
    main()