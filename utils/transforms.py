#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:22:10 2018
Transforms for visual ratings project
@author: gustav
"""

from __future__ import print_function, division
#import os
import torch

from skimage import io, transform
import numpy as np
#from scipy import misc,ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

import math

# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

class ComposeMRI(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input= t(input)   
        return input

class ComposeCT(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input, center_voxels):
        for t in self.transforms:
            
            try:
                input= t(input,center_voxels)
            except:
                input= t(input)
    
        return input
class SwapAxes(object):
    """Switch axes so that convolution is applied in axial plane

    Args:
        axis1: dim1 to be swapped with axis2
    """

    def __init__(self,axis1,axis2):
       self.axis1 =axis1
       self.axis2 =axis2
    def __call__(self, image):
        
        return np.swapaxes(image,self.axis1,self.axis2)

class ReturnAllPlanes(object):
    """
    Returns three images for each plane for PA model

    Args:
        axis1: dim1 to be swapped with axis2
        image is a tensor
    """

    #def __init__(self,axis1=0,axis2=1):
     #  self.axis1 =axis1
      # self.axis2 =axis2
    def __call__(self, image_1):
        image_2 = image_1.permute(1,0,2)
        image_3 = image_1.permute(2,1,0)
        
        
        # for 3d!!
        image_1 =image_1.unsqueeze(0)
        
        
        
        
        return image_1, image_2, image_3

class Return4D(object):
    """
    Returns 3D volume as a 4D object with 1 channel (i.e. dim: 1x h x w x d) to 
    work with PyTorch 3D functions

    Args:
        MRI volume
        image is a tensor
    """
    def __call__(self, img):
        
        img =img.unsqueeze(0)

        return img
    
class Return5D(object):
    """
    Returns three images for each plane for PA model

    Args:
        axis1: dim1 to be swapped with axis2
        image is a tensor
    """
    def __init__(self, nc=1):
        self.nc = nc
    def __call__(self, image_1):
        
        image_1 =image_1.unsqueeze(1)

        if self.nc !=1:
            image_1 = image_1.repeat(1,self.nc,1,1)
        
        return image_1

class Return5DAllPlanes(object):
    """
    Returns three images for each plane for recurrent PA model

    Args:
        nc - number of channels (default: 1)
        ax_lim - [slice_1,slice_n]. Only keep axial slices between slice_1 and slice_n.
        cor_lim - [slice_1,slice_n]. Only keep coronal slices between slice_1 and slice_n.
        sag_lim - [slice_1,slice_n]. Only keep sagittal slices between slice_1 and slice_n.
        stepsize - [ax_s,cor_s,sag_s]. List with 
        
    """
    #def __init__(self, nc=1,ax_lim=[4,-4], cor_lim=[26,134], sag_lim = [25,125],stepsize = [2,3,2]):
    def __init__(self, nc=1,ax_lim=[50,-5], cor_lim=[20,75], sag_lim = [30,-30],stepsize = [2,2,2]):
        
        self.nc = nc
        self.ax_lim = ax_lim
        self.cor_lim = cor_lim
        self.sag_lim = sag_lim
        self.stepsize = stepsize
    def __call__(self, ax):
        ax_s,cor_s,sag_s = self.stepsize
        sag = ax.permute(1,0,2)
        cor = ax.permute(2,1,0)
        
        ax = ax[self.ax_lim[0]:self.ax_lim[1],:,:]
        cor = cor[self.cor_lim[0]:self.cor_lim[1],:,:]
        sag = sag[self.sag_lim[0]:self.sag_lim[1],:,:]
        
        ax = ax[0::ax_s,:,:]
        cor = cor[0::cor_s,:,:]
        sag = sag[0::sag_s,:,:]
        if False:
            print(cor.shape)
            plt.figure()
            plt.imshow(cor.detach().numpy()[0,:,:])
            plt.show()
            
            plt.figure()
            plt.imshow(ax.detach().numpy()[20,:,:])
            plt.show()
            
            plt.figure()
            plt.imshow(sag.detach().numpy()[20,:,:])
            plt.show()
            input()
        ax =ax.unsqueeze(1)
        cor =cor.unsqueeze(1)
        sag =sag.unsqueeze(1)

        if self.nc !=1:
            ax = ax.repeat(1,self.nc,1,1)
            sag = sag.repeat(1,self.nc,1,1)
            cor = cor.repeat(1,self.nc,1,1)
        '''
        print('------')
        print(ax.shape)
        print(cor.shape)
        print(sag.shape)
        print('------')
        '''
        return ax,cor,sag

class ReturnStackedPA(object):
    """
    Returns three images for each plane for recurrent PA model

    Args:
        nc - number of channels (default: 1)
        ax_lim - [slice_1,slice_n]. Only keep axial slices between slice_1 and slice_n.
        cor_lim - [slice_1,slice_n]. Only keep coronal slices between slice_1 and slice_n.
        sag_lim - [slice_1,slice_n]. Only keep sagittal slices between slice_1 and slice_n.
        stepsize - [ax_s,cor_s,sag_s]. List with 
        
    """
    #def __init__(self, nc=1,ax_lim=[4,-4], cor_lim=[26,134], sag_lim = [25,125],stepsize = [2,3,2]):
    def __init__(self, nc=1,ax_lim=[50,-5], cor_lim=[20,75], sag_lim = [30,-30],stepsize = [2,2,2],rnn=True):
        
        self.nc = nc
        self.ax_lim = ax_lim
        self.cor_lim = cor_lim
        self.sag_lim = sag_lim
        self.stepsize = stepsize
        self.rnn=rnn
    def __call__(self, ax):
        ax_s,cor_s,sag_s = self.stepsize
        sag = ax.permute(1,0,2)
        cor = ax.permute(2,1,0)
        
        ax = ax[self.ax_lim[0]:self.ax_lim[1],:,:]
        cor = cor[self.cor_lim[0]:self.cor_lim[1],:,:]
        sag = sag[self.sag_lim[0]:self.sag_lim[1],:,:]
        
        ax = ax[0::ax_s,:,:]
        cor = cor[0::cor_s,:,:]
        sag = sag[0::sag_s,:,:]
        
        ax =ax.unsqueeze(1)
        cor =cor.unsqueeze(1)
        sag =sag.unsqueeze(1)

        if self.nc !=1:
            ax = ax.repeat(1,self.nc,1,1)
            sag = sag.repeat(1,self.nc,1,1)
            cor = cor.repeat(1,self.nc,1,1)
        img = torch.cat((ax,cor,sag),dim=0)

        if not self.rnn:
            img = img.squeeze(1)
            
        return img

class ReduceSlices(object):
    """Reduce the number of slices in all dimensions by selecting every factor_hw:th element
    in hw-dimensions and every factor_d:th element in depth dimension

    Args:
        factor_hw, factor_d_d (int): Desired reduction factor.
    """

    def __init__(self, factor_hw, factor_d):

        self.f_h = factor_hw
        self.f_w = factor_hw
        self.f_d = factor_d
        
    def __call__(self, image):
        #start_time = time.time()
        h, w, d = image.shape[:3]
        #print('-------')
        
        image = image[0::self.f_h,0::self.f_w,0::self.f_d]
        
        #end_time = time.time()
        #print('--- ReduceSlices: ' + str(end_time - start_time))
        return image

class Scale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w, d = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w, new_d = self.output_size

        new_h, new_w, new_d = int(new_h), int(new_w), int(new_d)

        #scale=(new_h*1.)/h

        print('Does not work!!! Scale()')
        img = transform.resize(image, (new_h))

        return img#{'image': img, 'landmarks': landmarks}
class CenterCropCT(object):
    """Crop center of the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, output_d):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.output_d = output_d
    def __call__(self, image,center_voxels):
        #image, landmarks = sample['image'], sample['landmarks']
        #start_time = time.time()
        h, w, d = image.shape[:3]
        new_h, new_w = self.output_size
        new_d = self.output_d
        hc,wc,dc = center_voxels
        top = max(0,int(round((2*hc-new_h)/2.)))
        if (top + new_h)>h:
            top=int(h-new_h)
        left = max(0,int(round((2*wc-new_w)/2.)))
        
        if (left+ new_w)>w:
            left=int(w-new_w)
        
        depth = max(0,int(round((2*dc-new_d)/2.)))
        
        if (depth + new_d)>d:
            depth=int(d-new_d)
        
        image = image[top: top + new_h,
                      left: left + new_w, depth:depth+new_d]
        return image.copy()
class Threshold(object):
    """Threshold numpy values in image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, lower_limit, upper_limit):
        
        self.ll = lower_limit
        self.ul = upper_limit
    def __call__(self, image):
        image[image<=self.ll]=self.ll
        image[image>=self.ul]=self.ul
        return image

class RandomNoise(object):
    """Add random normally distributed noise to image tensor.

    Args:
        noise_var (float): maximum variance of added noise 
        p (float): probability of adding noise
    """

    def __init__(self, noise_var=.1, p=.5):
        
        self.noise_var = noise_var
        self.p = p
    def __call__(self, image):
        if torch.rand(1)[0]<self.p:
            var = torch.rand(1)[0]*self.noise_var
            #plt.figure(figsize=(12,16))
            #plt.subplot(1,3,1)
            #img1 = image.detach()
            #plt.imshow(img1[10,:,:])
            image += torch.randn(image.shape)*var
            #plt.subplot(1,3,2)
            #plt.imshow(image[10,:,:])
            #plt.colorbar()
            #plt.subplot(1,3,3)
            #plt.imshow(image.detach()[10,:,:]-img1.detach()[10,:,:])
            #plt.show()
        return image

class CenterCrop(object):
    """Crop center of the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_x, output_y, output_z ,offset_x=0,offset_y = 0,offset_z=0):
        
        self.output_x = int(output_x)
        self.output_y = int(output_y)
        self.output_z = int(output_z)
        
        self.offset_x=int(offset_x)
        self.offset_y=int(offset_y)
        self.offset_z=int(offset_z)
    def __call__(self, image):
        
        img_min = image.min()
        img_max = image.max()
        img_dif = img_max-img_min
        image = image-img_min
        img_mean = image.mean()
        img_median = np.median(image)
        x_orig, y_orig, z_orig = image.shape[:3]
        #print([x_orig, y_orig, z_orig])
        x_mid = int(x_orig/2.); y_mid = int(y_orig/2.); z_mid = int(z_orig/2.);
        #print([self.output_x, self.output_y, self.output_z])
        new_x, new_y, new_z = self.output_x, self.output_y, self.output_z
        
        top = int(round((y_orig-new_y)/2.))
        left = int(round((x_orig-new_x)/2.))
        depth = int(round((z_orig-new_z)/2.))
        
        x=int(x_mid +self.offset_x - round(new_x/2.))
        y=int(y_mid +self.offset_y - round(new_y/2.))
        z=int(z_mid + self.offset_z - round(new_z/2.))

        dif = x+new_x-x_orig
        #print(dif)
        if dif>0:
            
            x=0; new_x = x_orig
        dif = y+new_y-y_orig
        if dif>0:
            #print('y')
            #y -= dif
            y=0; new_y = y_orig
        dif = z+new_z-z_orig
        if dif>0:
            #print('z')
            #z -= dif
            z=0; new_z = z_orig
        
        #print(image.shape)
        #print([new_x,new_y,new_z])
        #print([x,y,z])
        #print('----')
        image = image[x: x + new_x,
                      y: y + new_y, z:z+new_z]
        #print(image.shape)
        #image = (image-img_min)/(img_dif)        
        #image = image/img_mean*.11
        image = image/img_mean
        
        
        return image#.copy()

class PadOrCrop(object):
    """
    Pad (or crop if original size is larger than specified sizes) 
    input volume to given size. Assumes that 

    Args:
        output_sizes: 
            output_x,output_y,output_z 
    """

    def __init__(self, output_x, output_y, output_z,offset_x=0,offset_y = 0,offset_z=0):
        
        self.output_x = int(output_x)
        self.output_y = int(output_y)
        self.output_z = int(output_z)
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)
        self.offset_z = int(offset_z)
    def __call__(self, image):
        #image, landmarks = sample['image'], sample['landmarks']
        
        x, y, z = image.shape[:3]
        new_x, new_y, new_z = self.output_x, self.output_y, self.output_z
        cropper = CenterCrop(new_x,new_y,new_z,self.offset_x,self.offset_y,self.offset_z)
        #print(image.shape)
        image = cropper(image)
        #print(image.shape)
        x, y, z = image.shape[:3]
        
        if not [x,y,z]==[new_x,new_y,new_z]:

            image_tmp = np.zeros((new_x,new_y,new_z),dtype=np.float32)                
            
            six = []
            for i in range(len(image.shape)):
                six.append(int((image_tmp.shape[i]-image.shape[i])/2.))
            
            image_tmp[six[0]: six[0] + x,
                          six[1]: six[1] + y, six[2]:six[2]+z] = image
            image=image_tmp
        
        return image#.copy()#{'image': image, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_x, output_y, output_z):
        
        self.output_x = output_x
        self.output_y = output_y
        self.output_z = output_z
    def __call__(self, image):
        #image, landmarks = sample['image'], sample['landmarks']
        
        x, y, z = image.shape[:3]
        new_x, new_y, new_z = self.output_x, self.output_y, self.output_z

        x = np.random.randint(0, x - new_x)
        y = np.random.randint(0, y - new_y)
        z = np.random.randint(0, z - new_z)

        image = image[x: x + new_x,
                      y: y + new_y, z:z+new_z]
        return image#.copy()#{'image': image, 'landmarks': landmarks}

class RandomMirrorLR(object):
    """Randomly mirror an image in the sagittal plane (left-right)"""
    def __init__(self, axis):
        self.axis=axis
    def __call__(self, image):
        randbol = np.random.randn()>0 # 50/50 if to rotate
        #image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if randbol:#randbol:
        
            image = np.flip(image,self.axis).copy()
        
        return image    


class RotateVolume(object):
    def __init__(self, axis):
        self.axis=axis
    def __call__(self, img_rot):
        #img_rot = img.copy()
        # create meshgrid 
        if np.random.rand()<0.1: #only rotate 5% of time
            d1,d2,d3 = img_rot.shape
            #line1=;line2=;line3=
            coords=np.meshgrid(np.arange(d2), np.arange(d1),np.arange(d3))
            xyz=np.vstack([coords[0].reshape(-1) -d1/2., 
                           coords[1].reshape(-1)-d2/2.,
                           coords[2].reshape(-1)-d3/2.,
                           np.ones((d2,d1,d3)).reshape(-1)])
            #xyz=np.vstack([coords[0].reshape(-1), coords[1].reshape(-1),coords[1].reshape(-1)])
            # rotation matrix
        
            angle_deg =np.random.randn()*5
             
            angle=angle_deg/180.*np.pi
            #print(angle_deg)
            if self.axis==0:
                mat=rotation_matrix(angle,(1,0,0))
            elif self.axis==1:
                mat=rotation_matrix(angle,(0,1,0))
            elif self.axis==2:
                mat=rotation_matrix(angle,(0,0,1))
                
            xyz_rot=np.dot(mat, xyz)
            #xyz_rot = xyz
            x=xyz_rot[0,:]+float(d1)/2.
            y=xyz_rot[1,:]+float(d2)/2.
            z=xyz_rot[2,:]+float(d3)/2.
        
            x=x.reshape((d1,d2,d3))
            y=y.reshape((d1,d2,d3))
            z=z.reshape((d1,d2,d3))
        
            # the coordinate system seems to be strange, it has to be ordered like this
            new_xyz=[y,x,z]
        
            # sample
            img_rot=map_coordinates(img_rot,new_xyz, order=0)
        return img_rot
        
class PerImageNormalization(object):
    """ Transforms all pixel values to to have total mean= 0 and std = 1"""

    def __call__(self, image):
        # transform data to 
        
        image -=image.mean()
        image /=image.std()

        return image

class PrcCap(object):
    """ Cap all pixel values to prc 5,95 
    low: lower cap
    high: upper percentile value to cap high pixel values to.
    """
    def __init__(self, low=5,high=99):
        self.low = low
        self.high= high
    def __call__(self, image):
        # transform data to 
        
        l= np.percentile(image,self.low)
        h= np.percentile(image,self.high)
        image[image<l] = l; image[image>h] = h

        return image

# USE torchvision
class ToTensorFSL(object):
    """Convert np arrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        return image

