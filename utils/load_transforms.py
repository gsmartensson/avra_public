#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File contains a transform loader for train and validation set for simpler 
experiments, and consistency between training_script.py and avra.py transforms. 

"""
from torchvision.transforms import Compose
import utils.transforms as tfs
def compose_transform_parts(pre=None,mid=None,mid_2=None,post=None):
    '''
    Composes transform parts. Helps since some of the transform steps are shared by 
    the training and the validation dataset.
    '''
    tmp = [pre,mid,mid_2,post]
    parts=[]
    for part in tmp:
        if part is not None:
            parts.extend(part)
    
    return parts
def load_transform(args):
    '''
    Returns data transform for each rating scale.
    '''
    if args.vrs=='mta':
        pre = [tfs.SwapAxes(1,2)]
        mid_train = [tfs.CenterCrop(args.size_x+10,args.size_y+10,args.size_z+6, offset_x =args.offset_x, offset_y =args.offset_y, offset_z =args.offset_z),tfs.RandomCrop(args.size_x,args.size_y,args.size_z)]
        mid_train_2 = [tfs.ReduceSlices(1,2)]
        mid_test = [tfs.CenterCrop(args.size_x,args.size_y,args.size_z, offset_x =args.offset_x, offset_y =args.offset_y, offset_z =args.offset_z)]
        if args.arch=='VGG_bl': # for VGG baseline comparison in research paper
            post = [tfs.ToTensorFSL(),tfs.PerImageNormalization()]
        else:
            post = [tfs.ToTensorFSL(),tfs.PerImageNormalization(),tfs.Return5D(nc=args.nc)]
        
        transform_train = Compose(compose_transform_parts(pre=pre,mid=mid_train,post=post))
        transform_train_x = Compose(compose_transform_parts(pre=pre,mid=mid_train,mid_2=mid_train_2,post=post))
        transform_test = Compose(compose_transform_parts(pre=pre,mid=mid_test,post=post))
        
    elif args.vrs=='gca-f':
        pre = None
        mid_train = [#tf.RotateVolume(1),tf.RotateVolume(0),
                     tfs.CenterCrop(args.size_x+10,args.size_y+10,args.size_z+6,
                                   offset_x =args.offset_x, offset_y=args.offset_y, offset_z =args.offset_z),tfs.RandomCrop(args.size_x,args.size_y,args.size_z)]
        mid_train_2 = [tfs.ReduceSlices(1,2),tfs.RandomMirrorLR(0)]
        mid_train_2_x = [tfs.ReduceSlices(1,3),tfs.RandomMirrorLR(0)]
        mid_test = [tfs.CenterCrop(args.size_x,args.size_y,args.size_z,
                                   offset_x =args.offset_x, offset_y=args.offset_y, offset_z =args.offset_z),tfs.ReduceSlices(1,2)]
        if args.arch=='VGG_bl':
            post = [tfs.ToTensorFSL(),tfs.PerImageNormalization()]
        else:
            post_train = [tfs.ToTensorFSL(),tfs.PerImageNormalization(),tfs.RandomNoise(noise_var=.05,p=.5),tfs.Return5D(nc=args.nc)]
            post_test = [tfs.ToTensorFSL(),tfs.PerImageNormalization(),tfs.Return5D(nc=args.nc)]
            
        transform_train = Compose(compose_transform_parts(pre=pre,mid=mid_train,mid_2=mid_train_2,post=post_train))
        transform_train_x = Compose(compose_transform_parts(pre=pre,mid=mid_train,mid_2=mid_train_2_x,post=post_train))
        transform_test = Compose(compose_transform_parts(pre=pre,mid=mid_test,post=post_test))
        
    elif args.vrs=='pa':
        pre = None
        mid_train = [tfs.CenterCrop(args.size_x+10,args.size_y+10,args.size_z+6, offset_y=args.offset_y, offset_x=args.offset_x, offset_z=args.offset_z), tfs.RandomCrop(args.size_x,args.size_y,args.size_z)]
        mid_test = [tfs.CenterCrop(args.size_x,args.size_y,args.size_z, offset_y=args.offset_y, offset_x =args.offset_x, offset_z =args.offset_z)]
        if args.arch=='VGG_bl':
            post = [tfs.ToTensorFSL(),tfs.PerImageNormalization(),
                    tfs.ReturnStackedPA(nc=args.nc,rnn=False)]
        else:
            post = [tfs.ToTensorFSL(),tfs.PerImageNormalization(),
                    tfs.ReturnStackedPA(nc=args.nc)]
        transform_train = Compose(compose_transform_parts(pre=pre,mid=mid_train,post=post))
        transform_train_x = Compose(compose_transform_parts(pre=pre,mid=mid_train,post=post))
        transform_test = Compose(compose_transform_parts(pre=pre,mid=mid_test,post=post))
    return transform_train, transform_test,transform_train_x

