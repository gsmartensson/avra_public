# -*- coding: utf-8 -*-
"""
Misc. functions useful for training/evaluating/using models.

"""
from __future__ import division
#import csv
import torch
import torch.nn.functional as F
#from sklearn.metrics import cohen_kappa_score,confusion_matrix

import torch.nn.parallel
import torch.optim
import torch.utils.data
#import matplotlib.pyplot as plt
import numpy as np
import os
import nipype.interfaces.fsl as fsl
from collections import OrderedDict
from model.model import AVRA_rnn, VGG_bl
import nibabel

from shutil import copyfile
import glob
#import copy
def load_mri(file):
    # Load mri files with same orientation and numpy format
    a = nibabel.load(file) # load file
    a = nibabel.as_closest_canonical(a) # transform into RAS orientation
    a = np.array(a.dataobj,dtype=np.float32) # fetch data as float 32bit
    return a

def load_settings_from_model(path,args):
    # Loads settings used for during training AVRA, s.a. network architexture and preprocessing parameters (e.g. image dimensions)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    args.size_x,args.size_y, args.size_z = checkpoint['size_x'],checkpoint['size_y'],checkpoint['size_z']
    args.arch = checkpoint['arch']
    
    args.mse  =checkpoint['mse']
    if 'offset_x' in checkpoint.keys(): # offset_x, y, z not saved in AVRA versions < .7
        args.offset_x, args.offset_y, args.offset_z = checkpoint['offset_x'], checkpoint['offset_y'], checkpoint['offset_z']
        args.nc = checkpoint['nc']
    else: 
        args.nc = 1
        if args.vrs=='mta':
            args.offset_x ,args.offset_y, args.offset_z = 0, 0,4
        elif args.vrs =='gca-f':
            args.offset_x, args.offset_y, args.offset_z = 0, 5,14
        elif args.vrs =='pa':
            args.offset_x,args.offset_y,args.offset_z =0,-25,5
    return args

def load_model(path,args):
    # Load pretrained AVRA model and necessary parameters
    rating_scale = args.vrs
    if rating_scale=='mta':
        classes = [0,1,2,3,4]
    else:
        classes = [0,1,2,3]

    checkpoint = torch.load(path, map_location='cpu')

    #arch = checkpoint['arch']
    mse = checkpoint['mse']
    if mse:
        output_dim = 1
    else:
        output_dim = np.size(classes)
    try:
        # keys used in early stages of AVRA
        d = checkpoint['depth']
        h = checkpoint['width']
        x,y,z = h,h,d
    except:
        x = checkpoint['size_x']
        y = checkpoint['size_y']
    #    z = checkpoint['size_z']

    is_data_dp = False

    # load model
    model = AVRA_rnn([x,y,1]) # 1"=" process 1 slice/cnn

    # Load pretrained weights
    new_state_dict = OrderedDict()
    
    # if model was trained using dataparallel module
    for k, v in checkpoint['state_dict'].items():
        if k[:6]=='module':
            is_data_dp = True
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
     
    if is_data_dp:
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])
    return model.to(args.device)


def native_to_tal_fsl(path_to_img, force_new_transform=False,dof=6,output_folder = '/home/gustav/Desktop/data_processed/mta_data/native',guid='',remove_tmp_files=True):
    '''
    path_to_img - input image in native space
    force_new_transform - if True, the native image will be transformed regardless of if 
    a transformed image exists
    returns ac/pc alinged imaged by taliarach transformation with voxel size of 1x1x1mm3
    Function that inputs a native image of the brain and 
        1) conforms it to 1x1x1mm3 voxel size + lrflip if needed
        2) performs rigid talariach transformation for ac/pc-alignment and centering of brain.
        3) if to_nifti: convert to nii
    ''' 
    native_img = os.path.basename(path_to_img)
    if not guid: # if no output guid soecified
        if 'nii.gz' in native_img:
            guid = os.path.splitext(os.path.splitext(native_img)[0])[0]
        else:
            guid = os.path.splitext(native_img)[0] 
    tal_img = guid + '_mni_dof_'+str(dof) + '.nii'
    bet_img = guid + '_bet.nii'
    bet_img_cp = guid + '_bet_cp.nii'
    tmp_img = guid + '_tmp.nii'
    path_to_folder = os.path.dirname(path_to_img)
    tmp_img_path = os.path.join(output_folder,tmp_img)
    tal_img_path= os.path.join(output_folder,tal_img)
    bet_img_path= os.path.join(output_folder,bet_img)
    bet_img_path_cp= os.path.join(output_folder,bet_img_cp)

    xfm_path = os.path.join(output_folder,guid + '_mni_dof_' + str(dof)+ '.mat')
    xfm_path_cp = os.path.join(output_folder,guid + '_mni_dof_' + str(dof)+ '_cp.mat')
    
    xfm_path2 = os.path.join(output_folder,guid + '_mni_dof_' + str(dof)+ '_2.mat')
    
    try:
        fsl_path=os.environ['FSLDIR']
    except:
        fsl_path='/usr/local/fsl'
        print('please install fsl and test $FSLDIR. Trying default path: ' + fsl_path)
    template_img = os.path.join(fsl_path,'data','standard','MNI152_T1_1mm.nii.gz')
    tal_img_exist = os.path.exists(tal_img_path)
    xfm_exist = os.path.exists(xfm_path)
    fsl_1 = fsl.FLIRT()
    fsl_2 = fsl.FLIRT()
    fsl_pre = fsl.Reorient2Std()
    
    
    if not tal_img_exist or force_new_transform:
        
        # pre-reorient first images
        fsl_pre.inputs.in_file = path_to_img
        fsl_pre.inputs.out_file = tmp_img_path
        
        fsl_pre.inputs.output_type ='NIFTI'
        fsl_pre.run()
        
        # run skull strip to calculate transformation matrix
        btr = fsl.BET()
        btr.inputs.in_file = tmp_img_path
        btr.inputs.frac = 0.7
        btr.inputs.out_file = bet_img_path
        btr.inputs.output_type ='NIFTI'
        #btr.inputs.reduce_bias = True
        btr.inputs.robust = True
        btr.cmdline
        btr.run() 
        
        # calculate transformation matrix - 1st attempt
        fsl_1.inputs.in_file = bet_img_path
        fsl_1.inputs.reference = template_img
        fsl_1.inputs.out_file = bet_img_path
        fsl_1.inputs.output_type ='NIFTI'
        fsl_1.inputs.dof = dof
        fsl_1.inputs.out_matrix_file = xfm_path
        fsl_1.run()
        
        # read .mat file to assess if AC-PC alignment failed completely by looking at the diagonal elements (should be close to 1's)
        f = open(xfm_path, 'r')
        l = [[num for num in line.split('  ')] for line in f ]
        matrix_1 = np.zeros((4,4))
        for m in range(4):
            for n in range(4):
                matrix_1[m,n] = float(l[m][n])
        
        dist_1 = np.sum(np.square(np.diag(matrix_1)-1))
        print('Dist (below 0.01 generally OK): ' + str(dist_1))
        print('Transformation matrix path: ' + xfm_path)
        dist_lim = .01
        translate_lim  = 30
        if dist_1>dist_lim or matrix_1[2,3]>translate_lim : # if ac-PC failed, run without bet
            # copy bet files for debuging                
            copyfile(bet_img_path, bet_img_path_cp)
            copyfile(xfm_path, xfm_path_cp)
            
            print('------ Rerunning registration without bet ---')
            fsl_1.inputs.in_file = tmp_img_path
            fsl_1.run()
        
            f = open(xfm_path, 'r')
            l = [[num for num in line.split('  ')] for line in f ]
            matrix_2 = np.zeros((4,4))
            for m in range(4):
                for n in range(4):
                    matrix_2[m,n] = float(l[m][n])
            
            dist_2 = np.sum(np.square(np.diag(matrix_2)-1))
            print([dist_1,dist_2])
            if (dist_1<dist_lim and dist_2<dist_lim):
                if matrix_1[2,3]<matrix_2[2,3]:
                    # use the transform from the bet image if that was "better"
                    xfm_path=xfm_path_cp
                    print('Using bet transform, both below dist and translate smaller')
            elif dist_1<dist_2:
                xfm_path=xfm_path_cp
                print('Using bet transform, bet dist smaller')
        
        # apply transform
        fsl_2.inputs.in_file = tmp_img_path
        fsl_2.inputs.reference = template_img
        fsl_2.inputs.out_file = tal_img_path
        fsl_2.inputs.output_type ='NIFTI'
        fsl_2.inputs.in_matrix_file = xfm_path
        fsl_2.inputs.apply_xfm = True
        fsl_2.inputs.out_matrix_file = xfm_path2     
        fsl_2.run()
        if remove_tmp_files:
            for img in [tmp_img_path,bet_img_path,xfm_path2, bet_img_path_cp,xfm_path_cp]:#,xfm_path
                if os.path.exists(img):
                    os.remove(img)
        else:
            print('OBS: NOT REMOVING TEMPORARY TRANSFORM FILES ')
