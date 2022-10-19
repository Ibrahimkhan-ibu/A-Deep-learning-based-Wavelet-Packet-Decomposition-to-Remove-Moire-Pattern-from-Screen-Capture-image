#Pytorch libraries
import torchvision.models.vgg as vgg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models.vgg as vgg
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.utils.data import Dataset
from torch.utils import data


#Other libraries
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image
import pdb
import pickle
import matplotlib 
matplotlib.rcParams['backend'] = "Agg" 
import random












#variables)
epoch_start= 0
target_epochs=50
saved_model_folder_name= 'moire_weights'
batch_length=10
cpu_cores=10
cuda = True if torch.cuda.is_available() else False #check if cuda toolkit is available or not





#source: https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose     
mytransform = transforms.Compose([    
     transforms.ToTensor(),   
    ])

myfolder = myImageFloder(root = '/home/ibrahim/thesis/datasets/trainData',  transform = mytransform)
dataloader = DataLoader(myfolder, num_workers=cpu_cores, batch_size=batch_length, shuffle=True)
print('DATA PRE-PROSESSING COMPLETE')










#l1 loss between the image and its ground-truth in the RGBdomain,
def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input-output))
 







'''
SOURCE:
Johnson, J., Alahi, A., &amp; Fei-Fei, L. (1970, January 1). Perceptual losses for real-time style transfer and Super-Resolution. SpringerLink. 
Retrieved September 29, 2022, from https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43 
https://deepimagination.cc/docs/_modules/imaginaire/losses/perceptual.html
CODE:
https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer
https://github.com/rajeevyasarla/Syn2Real/blob/master/perceptual.py
'''
class LossNetwork(torch.nn.Module):



    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",        #1_2 to 5_2
        }
        
    def forward(self, x):
        output = {}
        #import pdb
        #pdb.set_trace()
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        
        return output

        
 
























# Loss functions

MAE = torch.nn.L1Loss() #Mean Absolute Error 
 













#Dialation from : Understanding convolution for semantic segmentation.
# Our proposed DeepWPD 
class DeepWPD(nn.Module): 
    def __init__(self,in_channel=3):
        super(DeepWPD,self).__init__() 

        self.cascade01=nn.Sequential(
            Conv2d(48, 64 , 1 , stride=1, padding=0), 
           
            nn.LeakyReLU(0.2, inplace=True) 

        )

        self.cascade02=nn.Sequential( 
                                    
            DRDN(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',delia=1),
            
            DRDN(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',delia=2),
            
            DRDN(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',delia=5),
            
            DRDN(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',delia=7),
            
            DRDN(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',delia=12),
            
            DRDN(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',delia=19),
            
            DRDN(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',delia=31)
        )
        
        self.final=nn.Sequential(
            conv_block(64,48, kernel_size=1, norm_type=None, act_type=None) 
        )
        
        
    def forward(self, x):
        x1 = self.cascade01(x)

        
        x2 = self.cascade02(x1)

        x3 = self.final(x2)
        
        return x3


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

      
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)





def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
     
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer



class DRDN(nn.Module):
    """
    Directional Residual Dense Network
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=1):
        super(DRDN, self).__init__()
        self.RDB01 = ResidualDenseBlock(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB02 = ResidualDenseBlock(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)

        self.dilation1 = nn.Sequential(
            Conv2d(64, 64 , 3, stride=1, dilation=delia),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dilation2 = nn.Sequential(
            Conv2d(64, 64 , 3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):

        output1 = self.RDB01(x)
        output2 = output1+x
        output3 = self.RDB02(output2)

        #Dialation Convolution Network (DCN)
        output4 = self.dilation1(x)+0.2*self.dilation2(self.dilation1(x))
        return output3.mul(0.2)+ output4


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block
    Source: Residual Dense Network for Image Super-Resolution, CVPR 18)
    CODE: https://github.com/yjn870/RDN-pytorch
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv01 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv02 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv03 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv04 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv05 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode=mode)
        

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv02(torch.cat((x, x1), 1))  
        x3 = self.conv03(torch.cat((x, x1, x2), 1))
        x4 = self.conv04(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv05(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:                                           #In our case it is 'CNA', norm_type=None, 
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


main_network = DeepWPD()





































"""
From Huang, Huaibo, et al. “Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination - 
International Journal of Computer Vision.” SpringerLink, Springer US, 12 Feb. 2019, https://link.springer.com/article/10.1007/s11263-019-01154-8. 
git: https://github.com/hhb072/WaveletSRNet 
"""
class WPT(nn.Module): #Wavelet Packet Transform 
    def __init__(self, scale=1, dec=True, params_path='/home/ibrahim/thesis/deepWPD_code/wavelet_weights_c2.pkl', transpose=True):
        super(WPT, self).__init__()
        
        self.scale = scale 
        self.dec = dec
        self.transpose = transpose
        
        ks = int(math.pow(2, self.scale)  )
        nc = 3 * ks * ks
        
        if dec:
          self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        else:
          self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path,'rb') 
                u = pickle._Unpickler(f) 
                u.encoding = 'latin1' 
                dct = u.load().
                f.close() 
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        if self.dec:
          output = self.conv(x)          
          if self.transpose:
            osz = output.size()
            output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)            
        else:
          if self.transpose:
            xx = x
            xsz = xx.size()
            xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)         
          output = self.conv(xx)        
        return output 






decomposition = WPT(scale=2, dec=True) 
WIT = WPT(scale=2, dec=False) #wavelet inverse transforme       
















if cuda:
    decomposition = decomposition.cuda()
    WIT = WIT.cuda() #wavelet inverse transform
    main_network = main_network.cuda()
    MAE=MAE.cuda()
    loss_network = LossNetwork().float().cuda()














def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.constant_(m.weight.data, 0.0)
        b,c,w,h = m.weight.shape
        cx, cy = w//2, h//2
        torch.nn.init.eye_(m.weight.data[:,:,cx,cy])
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if epoch_start != 0:
    main_network.load_state_dict(torch.load('/home/ibrahim/thesis/deepWPD_code/saved_models/moire_weights/lastest.pth' ))

else:
    # Initialize weights
    main_network.apply(init_weights)
    
device = torch.device("cuda:0")










# Optimizers
optimizer = torch.optim.Adam(main_network.parameters(), lr=0.0002, betas=(0.5, 0.999)) #From https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

 








  

'''
DATASET AND DATA PREPROSESSING
Source: 
Sun, Y., Yu, Y., &amp; Wang, W. (2018, May 8). 
Moiré photo restoration using multiresolution convolutional neural networks. arXiv.org. Retrieved October 1, 2022, from https://arxiv.org/abs/1805.02996 
CODE:
https://github.com/ZhengJun-AI/MoirePhotoRestoration-MCNN
'''
# change the root to your own data path

def default_loader(path1,path2): 
    #pdb.set_trace()
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    w,h=img1.size #width and hight

    # demoire photo dataset
    i = random.randint(-6,6)
    j = random.randint(-6,6)
    img1=img1.crop((int(w/6)+i,int(h/6)+j,int(w*5/6)+i,int(h*5/6)+j))
    img2=img2.crop((int(w/6)+i,int(h/6)+j,int(w*5/6)+i,int(h*5/6)+j))
    

    img1 = img1.resize((256,256),Image.BILINEAR )
    img2 = img2.resize((256,256),Image.BILINEAR )
    
    r= random.randint(0,1)
    if r==1:
        img1=img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2=img2.transpose(Image.FLIP_LEFT_RIGHT)
        
    t = random.randint(0,2)
    if t==0:
        img1=img1.transpose(Image.ROTATE_90)
        img2=img2.transpose(Image.ROTATE_90)
    elif t==1:
        img1=img1.transpose(Image.ROTATE_180)
        img2=img2.transpose(Image.ROTATE_180)
    elif t==2:
        img1=img1.transpose(Image.ROTATE_270)
        img2=img2.transpose(Image.ROTATE_270)

    return img1 ,img2

class myImageFloder(data.Dataset):
    def __init__(self,root,transform = None,target_transform = None,loader = default_loader):

        imgin = []
        imgout = []
        imgin_names = []
        imgout_names = []

        for img_name in os.listdir(os.path.join(root,'source')):
            if img_name !='.' and img_name !='..':
                imgin_names.append(os.path.join(root,'source',img_name))
                
        for img_name in os.listdir(os.path.join(root,'target')):
            if img_name !='.' and img_name !='..':
                imgout_names.append(os.path.join(root,'target',img_name))
        imgin_names.sort()
        imgout_names.sort()
        print(len(imgin_names),len(imgout_names))

        assert len(imgin_names)==len(imgout_names)
        self.root = root
        self.imgin_names = imgin_names
        self.imgout_names = imgout_names
        self.transform = transform
        self.loader = loader

    def __getitem__(self,index):
        imgin = self.imgin_names[index]
        imgout = self.imgout_names[index]
        
        img1,img2 = self.loader(imgin,imgout)
        
        if self.transform is not None:
            #pdb.set_trace()
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2

    def __len__(self):
        return len(self.imgin_names)












    























# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor





"""
From Huang, Huaibo, et al. “Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination - 
International Journal of Computer Vision.” SpringerLink, Springer US, 12 Feb. 2019, https://link.springer.com/article/10.1007/s11263-019-01154-8. 
git: https://github.com/hhb072/WaveletSRNet 
"""  


step = 0
for epoch_start in range(epoch_start, target_epochs): 
    for i, batch in enumerate(dataloader): 
        step = step+1
        
        
        latest_learning_rate = 0.0002*(1/2)**(step/100000) #learning rate
        for param_group in optimizer.param_groups: #Adam optimizer
            param_group["lr"] = latest_learning_rate

        
        '''
        data,pred,label
        real_A = data or moire image = moire_image
        real_B =  label or ground truth = ground_image
        fake_B = demoired image with DeepWPD = demoired_image
        '''

        # Model inputs
        img_train = batch
        moire_image, ground_image = Variable(img_train[0].cuda()), Variable(img_train[1].cuda()) 
 
        x_r = (moire_image[:,0,:,:]*255-105.648186)/255.+0.5  
        x_g = (moire_image[:,1,:,:]*255-95.4836)/255.+0.5  
        x_b = (moire_image[:,2,:,:]*255-86.4105)/255.+0.5 
        moire_image = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1) 
                                                                                        
  
        y_r = ((moire_image[:,0,:,:]-0.5)*255+121.2556)/255.
        y_g = ((moire_image[:,1,:,:]-0.5)*255+114.89969)/255.
        y_b = ((moire_image[:,2,:,:]-0.5)*255+102.02478)/255.
        moire_image = torch.cat([ y_r.unsqueeze(1) , y_g.unsqueeze(1) , y_b.unsqueeze(1)  ],1)
        
        
        target_wavelets = decomposition(ground_image) 
        wavelets_lr_ground = target_wavelets[:,0:3,:,:] #LR stands for Long Reach or low-frequency components
        wavelets_sr_ground = target_wavelets[:,3:,:,:] #SR stands for Short Reach or high-frequency components
        
        source_wavelets = decomposition(moire_image)
        
        if epoch_start >-1 :
            
        
            optimizer.zero_grad()

        
            tensor_c = torch.from_numpy(np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1,3,1,1))).cuda()
           
            
            
            wavelets_demoired_image_re = main_network(source_wavelets)


            
            demoired_image = WIT(wavelets_demoired_image_re) +  moire_image       
            
            wavelets_demoired_image   = decomposition(demoired_image)
            wavelets_lr_demoired_image = wavelets_demoired_image[:,0:3,:,:]# LR stands for Long Reach or low-frequency components
            wavelets_sr_demoired_image= wavelets_demoired_image[:,3:,:,:]# SR stands for Short Reach or high-frequency components
            
       
            # Pixel-wise loss
            loss_pixel = MAE(demoired_image, ground_image)  


            # preceptual loss
            loss_demoired_image = loss_network(demoired_image*255-tensor_c)
            loss_ground_image = loss_network(ground_image*255-tensor_c)
            perceptual_1=compute_l1_loss(demoired_image*255-tensor_c,ground_image*255-tensor_c)*2
            perceptual_2=compute_l1_loss(loss_demoired_image['relu1'],loss_ground_image['relu1'])/2.6
            perceptual_3=compute_l1_loss(loss_demoired_image['relu2'],loss_ground_image['relu2'])/4.8
            total_perceptual_loss = perceptual_1+perceptual_2+perceptual_3   
            
           
            loss_lr = compute_l1_loss(wavelets_lr_demoired_image,  wavelets_lr_ground ) # loss MAE for SR or Short Reach or high-frequency components.
            loss_sr = compute_l1_loss(wavelets_sr_demoired_image,  wavelets_sr_ground ) # loss MAE for LR or Long Reach or low-frequency components.
            loss_wavelet=loss_sr.mul(100) + loss_lr.mul(10)


            
            
            
            





            '''
            Main LOSS FUNCTION
            
            SOURCE: From Huang, Huaibo, et al. “Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination - 
            International Journal of Computer Vision.” SpringerLink, Springer US, 12 Feb. 2019, https://link.springer.com/article/10.1007/s11263-019-01154-8. 
            CODE: https://github.com/hhb072/WaveletSRNet 
            '''


           loss_G = loss_pixel+ total_perceptual_loss + loss_wavelet  



  
        








            loss_G.backward() 




            optimizer.step()
           
            

            if i%100==0:
                sys.stdout.write("\r[epoch_start %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f] ETA: %s" %
                                                        (epoch_start, target_epochs,
                                                        i, len(dataloader),
                                                         loss_G.item(),
                                                        loss_pixel.item())) 
            
                
                
        else:
            pass;
            
            
    torch.save(main_network.state_dict(),'/home/ibrahim/thesis/deepWPD_code/saved_models/%s/lastest.pth'%saved_model_folder_name)
    
    
