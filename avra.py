#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Gustav Mårtensson 2018-2019

AVRA - Automatic Visual Ratings of Atrophy using recurrent convolutional neural networks. The script briefly:
    1. Takes a single unprocessed T1-weighted native MRI image in .nii.gz or .nii format.
    2. Uses FSL Flirt for a rigid registration (AC-PC alignment) and interpolation 
    to 1x1x1mm3 voxel size.
    3. An ensemble of neural networks (AVRA) predicts the scores of Scheltens' MTA scale (left and right), 
    Pasquier's frontal subscale of global cortical atrophy GCA-F, and Koedam's PA scale.
    4. A .csv file 'uid'.csv is created with the computed ratings. The ratings of interest are the mean
    output of the ensemble classifiers, e.g. pa_mean. The other values are for debugging purposes only.

AVRA was trained on ratings made by Neuroradiologist Lena Cavallin on images from ADNI1 and the Memory Clinic in Karolinska hospital.

Method and results are detailed in paper: https://www.sciencedirect.com/science/article/pii/S2213158219302220

To run:
python3 avra.py  --input-file /path/to/img.nii.gz --uid output_file_name_prefix --model-dir /path/to/trained/model/weights --output-dir /path/to/output/dir

"""
import csv
import torch
import argparse
import os 
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.parallel
import torch.optim
import torch.utils.data

import glob
import datetime
import collections

global args
from utils.load_transforms import load_transform
from utils.misc import load_model, load_settings_from_model,load_mri, native_to_tal_fsl


parser = argparse.ArgumentParser(description='AVRA: automatic visual ratings of atrophy using Recurrent Convolutional Neural Networks')
parser.add_argument('--model-dir', type=str,default='', help='Path to directory containing the folders trained network weights for each rating scale, e.g. model-dir/mta/')
parser.add_argument('--input-file', default='', help='Absoulute path to the input MRI file in (file assumed to be in .nii or .nii.gz format)')
parser.add_argument('--output-dir', default='', help='Path to directory where all output files. Directory will be created if it doesn\'t exist')
parser.add_argument('--uid', default='', type=str,help='Chosen unique id for output files that will located at output-dir/{uid_mni_dof_6.nii,uid.csv,uid_coronal.jpg}')
parser.add_argument('--no-new-registration', dest='registration', action='store_false',help='If a previous AC/PC-alignment exists (file output_folder/uid_mni_dof_6.nii) then setting this flag will use previous registration. If there is no previious transform, the transform will be performed anyway.')
parser.set_defaults(registration=True)
args = parser.parse_args()

print('---- Started automatic predictions of visual ratings using AVRA ----')
print('Input file: ' + args.input_file)
print('Output files: ' + args.output_dir + '/'+args.uid+'*')
print('Model directory: ' + args.model_dir)
print('Force new registration: ' + str(args.registration))

args.device =  torch.device('cpu')
timestamp= '{:%Y-%m-%d_%H_%M_%S}'.format(datetime.datetime.now())
fname= os.path.join(args.output_dir,args.uid + '_info.log') 

# Check that input parameters are OK
assert os.path.exists(args.input_file), 'input-file not specified or does not exist'
assert args.output_dir, 'output-dir not specified'
assert '.nii' in os.path.basename(args.input_file), 'input-file %s should be in .nii or .nii.gz format' % args.input_file
if not args.uid:
    args.uid=os.path.basename(args.input_file)
    args.uid = args.uid[:args.uid.find('.nii')]
    print('uid not specified. Automatically setting uid to %s' % args.uid)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print('Creating output-dir')

# Perform registration using FSL
if args.registration:
    print('Performing rigid registration of input image to MNI template (AC-PC alignment)')
else:
    print('If registration not found rigid registration of input image to MNI template (AC-PC alignment) is performed, else use previous registration')
dof=6 # degrees of freedom of transform
native_to_tal_fsl(args.input_file, force_new_transform=args.registration,dof=dof,output_folder =args.output_dir,guid=args.uid)
tal_path = os.path.join(args.output_dir,args.uid + '_mni_dof_'+str(dof) +'.nii')

# Log parameters in output csv file
rating_dict = collections.OrderedDict()
rating_dict['uid'] = args.uid
rating_dict['model-dir'] = args.model_dir

def print_results(vrs,output_vec):
    print('\n')
    print('-'*60)
    print('-'*27 + ' ' +vrs + ' ' +'-'*(26 - len(vrs)+5))
    print('-'*60)
    #print('\n')
    with np.printoptions(precision=3, suppress=True):
        print('Rating from each ensemble model: ' + str(output_vec))
    print('Mean AVRA rating: %.3f' % output_vec.mean())
    print('Median AVRA rating: %.3f' % np.median(output_vec))
    print('Std of ensemble ratings: %.3f' % output_vec.std())

with open(fname, 'w') as f:
    f.write(timestamp + '\n')
    f.write(str(args))

# Start automatic rating of PA, MTA (left and right) and GCA-F
for args.vrs in ['mta','pa','gca-f']:    
    # Load image
    img = load_mri(tal_path)
    
    # Load pretrained models, model parameters and transforms
    fnames = np.sort(glob.glob(args.model_dir+'/' +args.vrs+'/*.pth.tar'))
    assert len(fnames)>0, 'No pretrained model weights found in model-dir %s' % (args.model_dir)
    args = load_settings_from_model(fnames[0],args)
    transform_train, transform,_= load_transform(args)
    
    model_list = [load_model(ch,args) for ch in fnames]
    
    # MTA left and right
    if args.vrs=='mta':
        img =transform(img)
        sides = ['right','left']#,'left']            
        
        # Copy mirrored image for both MTA left and right rating        
        img_left = img.numpy().copy()  
        img_left = torch.from_numpy(np.flip(img_left,2).copy())

        img = img.unsqueeze(0)
        img_left = img_left.unsqueeze(0)
        imgs = [img,img_left]
         
        # Rate left and right MTA        
        for i, side in enumerate(sides):
            output_vec = []
            # Loop over trained ensemble models
            for m,model in enumerate(model_list):
                model.eval()
                
                output = model(imgs[i])
                output = output.mean()                
                
                output_vec.append(output.detach().numpy().squeeze())
                rating_dict[args.vrs +'_'+side +'_model_' + str(m+1)]=output_vec[-1]
            output_vec = np.array(output_vec)
            
            rating_dict[args.vrs +'_'+side +'_mean']=output_vec.mean()
            rating_dict[args.vrs +'_'+side +'_median']=np.median(output_vec)
            rating_dict[args.vrs +'_'+side +'_std']=output_vec.std()
            print_results(args.vrs +'_'+side,output_vec)
        # Plot coronal image for quality control puprpose
        # if saved image is not in the coronal plane, close to the MTA rating 
        # slice the FSL registration failed and automatic ratings can't be used
        mta_l = '{0:.2f}'.format(rating_dict[args.vrs +'_'+'left' +'_mean'])
        mta_r = '{0:.2f}'.format(rating_dict[args.vrs +'_'+'right' +'_mean'])

        fig = plt.figure(figsize=(8,8))
        plt.imshow(np.rot90(img_left[0,10,0,:,:]),cmap='gray')
        plt.axis('off')
        
        plt.title('MTA (right): ' + str(mta_r) +' - MTA (left): ' + str(mta_l))
        coronal_name = os.path.join(args.output_dir,args.uid + '_coronal.jpg')
        fig.savefig(coronal_name, format='jpg')
        plt.cla();plt.clf(); plt.close()

    # GCA-F and PA scales
    elif args.vrs=='gca-f' or args.vrs=='pa':
        img =transform(img)
        img = img.unsqueeze(0)
        img.to(args.device)
        output_vec = []
        
        # Loop over trained ensemble models
        for m,model in enumerate(model_list):
            model.eval()
            model.to(args.device)
            output = model(img)
            output =output.mean()
            output_vec.append(output.detach().numpy().squeeze())
            rating_dict[args.vrs +'_model_' + str(m+1)]=output_vec[-1]
        output_vec = np.array(output_vec)
        
        rating_dict[args.vrs +'_mean']=output_vec.mean()
        rating_dict[args.vrs +'_median']=np.median(output_vec)
        rating_dict[args.vrs +'_std']=output_vec.std()
        print_results(args.vrs,output_vec)
    
    # log model architecture and args used for automatic ratings
    with open(fname, 'a') as f:
        f.write('\n------ ' + args.vrs + ' -------\n')
        f.write(str(args))
        f.write(str(model))
        f.write(str(fnames))
        
# write ratings to csv file
csv_name = os.path.join(args.output_dir,args.uid + '.csv')    
with open(csv_name,'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=rating_dict.keys())
    writer.writeheader()
    writer.writerow(rating_dict)

print('-'*40)
print('Ratings written to ' + csv_name)
print('Inspect image to assure it is in the coronal plane close to the MTA rating slice: ' + coronal_name)
print('AVRA done!')
