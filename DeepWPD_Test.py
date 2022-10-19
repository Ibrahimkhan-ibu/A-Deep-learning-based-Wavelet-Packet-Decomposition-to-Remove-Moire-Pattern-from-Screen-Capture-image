import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torchvision.models.vgg as vgg
import pdb
from torchvision import transforms
from skimage import measure
from skimage import color


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






criterion_GAN = torch.nn.MSELoss()


'''
PSNR calculation
Source: Moire Photo Restoration Using Multiresolution 
CODE: https://github.com/ZhengJun-AI/MoirePhotoRestoration-MCNN/blob/master/test.py
'''
def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.double) - im2.astype(np.double)) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return psnr   




class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

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
        #import pdb
        #pdb.set_trace()
        return output
        
transform1 = transforms.Compose([
      transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
      #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
      ])       
def predict_img(net,
                img,
                lossnet,
                use_gpu=True):

    img=transform1(img)

    net.eval()
    lossnet.eval()
    
    w,h = img.shape[1],img.shape[2]
    tensor_c = torch.from_numpy(np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1,3,1,1)))
    real_a_pre = lossnet((img*255-tensor_c).cuda()) 
    
    relu_1 = nn.functional.interpolate(real_a_pre['relu1'].detach(),size=(w,h))
    relu_2 = nn.functional.interpolate(real_a_pre['relu2'].detach(),size=(w,h))
    relu_3 = nn.functional.interpolate(real_a_pre['relu3'].detach(),size=(w,h))
   
    precept = torch.cat([relu_1/255.,relu_2/255.,relu_3/255.],1)#,relu_4/255.,relu_5/255.], 1)
    
    img=img.unsqueeze(0)

    x_r = (img[:,0,:,:]*255-105.648186)/255.+0.5
    x_g = (img[:,1,:,:]*255-95.4836)/255.+0.5
    x_b = (img[:,2,:,:]*255-86.4105)/255.+0.5
    img = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1)
  
    y_r = ((img[:,0,:,:]-0.5)*255+121.2556)/255.
    y_g = ((img[:,1,:,:]-0.5)*255+114.89969)/255.
    y_b = ((img[:,2,:,:]-0.5)*255+102.02478)/255.
    img = torch.cat([ y_r.unsqueeze(1) , y_g.unsqueeze(1) , y_b.unsqueeze(1)  ],1)
    if use_gpu:
        img = img.cuda()
        net = net.cuda()
        
    with torch.no_grad():
        imgin = wavelet_dec(img)
        imgout = net(Variable(imgin))
        imgout =wavelet_rec(imgout) + img
        imgout = imgout.squeeze(0)

    return imgout



crit = criterion_GAN.cuda() 



input_path = '/home/ibrahim/thesis/dataset10w/test'
model= '/home/ibrahim/thesis/deepWPD_code/saved_models/moire_weights/lastest.pth'
viz=True



def get_output_filenames():
    in_files = input_path
    out_files = []
    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))


def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()

if __name__ == "__main__":
    in_files = 'input' #noise
    in_files2 = 'output' #original
    out_files = get_output_filenames()

    net = DeepWPD()
    lossnet= LossNetwork()
    wavelet_dec = WPT(scale=2, dec=True)
    wavelet_rec = WPT(scale=2, dec=False)        

    
    print("Begin Loading model {}".format(model))


    print("Using CUDA version of the net, prepare your GPU !")
    net.cuda()
    

    net.load_state_dict(torch.load(model))
    lossnet.cuda()
    wavelet_dec.cuda()
    wavelet_rec.cuda()






    print("Model loaded !")
    
    root = '/home/ibrahim/thesis/dataset10w/test' 
    
    im_files=os.listdir(os.path.join(root,in_files))  #noise images
    im_files2=os.listdir(os.path.join(root,in_files2)) #original images
    im_files.sort()
    im_files2.sort()

    
    log=open('result.txt','w') #save psnr&ssim values
    psnr_ori=0 #original image psnr
    psnr_pro=0 #demoired image psnr
    ssim_ori=0 #original image ssim
    ssim_pro=0 #demoired image ssim
    for i, fn in enumerate(im_files):
        
        print("\nprocessing image {} ...".format(fn))

        #noise images
        img = Image.open(os.path.join(root,in_files,fn))
        [w,h]=img.size
        img=img.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
        img = img.resize((256, 256),Image.BILINEAR)
        imgcom = img #noise images


        #original images
        imggt = Image.open(os.path.join(root,in_files2,im_files2[i]))
        imggt=imggt.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6))) #im.crop((left, upper, right, lower))
        imggt=imggt.resize((256,256),Image.BILINEAR)
        imggt = transform1(imggt) #original images



        #demoired image
        img2 = predict_img(net=net,
                           img=imgcom,
                           lossnet=lossnet,
                           use_gpu= True
                           )
        
        mse = crit(img2,imggt.cuda()) #MSELoss
        
        
        #demoired image
        img2 = (img2)*255
        img2=torch.clamp(img2,0,255);
        img2= np.uint8(img2.data.cpu().numpy())
        img2= img2.transpose((1,2,0)) 
        img2= Image.fromarray(img2)
        

        #original images
        imggt = (imggt)*255
        imggt=torch.clamp(imggt,0,255);
        imggt= np.uint8(imggt.data.cpu().numpy())
        imggt= imggt.transpose((1,2,0))  
        imggt= Image.fromarray(imggt)


        
        img_luma = color.rgb2ycbcr(np.array(img)[:,:,:])[..., 0] #noise images luma
        imggt_luma = color.rgb2ycbcr(np.array(imggt)[:,:,:])[..., 0]#original images luma
        img2_luma = color.rgb2ycbcr(np.array(img2)[:,:,:])[..., 0]#demoired image luma 

        po=cal_psnr(np.array(imggt_luma),np.array(img_luma)) #psnr_ori for each image
        p=cal_psnr(np.array(imggt_luma),np.array(img2_luma)) #psnr_pro for each image
        
        so=measure.compare_ssim(np.array(imggt_luma),np.array(img_luma),data_range=255)#ssim_ori for each image
        s=measure.compare_ssim(np.array(imggt_luma),np.array(img2_luma),data_range=255)#ssim_pro for each image

        psnr_ori=psnr_ori+po #total psnr_ori
        psnr_pro=psnr_pro+p #total psnr_pro
        
        ssim_ori=ssim_ori+so #total ssim_ori
        ssim_pro=ssim_pro+s #total ssim_pro
        
        
        print('psnr_ori:%f , psnr_pro:%f , ssim_ori:%f , ssim_pro:%f  mse:%f' % (po, p,so, s,mse))
        log.write('%d: psnr_ori:%f , psnr_pro:%f , ssim_ori:%f , ssim_pro:%f  mse:%f\n' % (i,po, p,so, s,mse))
        log.write('%s\n'%fn)
        if viz:
            h,w=img2.size
            imgout = np.zeros((w,3*h,3))
            imgout[0:w,0:0+h]=np.array(img)
            imgout[0:w,h:h+h]=np.array(img2)
            imgout[0:w,h*2:h*2+h]=np.array(imggt)
            imgout = Image.fromarray(imgout.astype(np.uint8))
            imgout.save('./testresult/output_%4d.jpg'%i)

        
    psnr_ori=psnr_ori/len(im_files)
    psnr_pro=psnr_pro/len(im_files)
    ssim_ori=ssim_ori/len(im_files)
    ssim_pro=ssim_pro/len(im_files)
    print('psnr_ori:%f , psnr_pro:%f , ssim_ori:%f , ssim_pro:%f' % (psnr_ori, psnr_pro,ssim_ori, ssim_pro))
    log.close()
