#!/usr/bin/env python
# coding: utf-8

# # Unseen Object Instance Segmentation

# In[ ]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import sys
import json
from time import time
import glob
from pathlib import Path


import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from rich import inspect

# My libraries. Ugly hack to import from sister directory
import src.data_augmentation as data_augmentation
import src.segmentation as segmentation
import src.evaluation as evaluation
import src.util.utilities as util_
import src.util.custom_utils as cu
import src.util.flowlib as flowlib


# ## Depth Seeding Network Parameters

# In[ ]:


dsn_config = {
    
    # Sizes
    'feature_dim' : 64, # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters' : 10, 
    'epsilon' : 0.05, # Connected Components parameter
    'sigma' : 0.02, # Gaussian bandwidth parameter
    'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
    'subsample_factor' : 5,
    
    # Misc
    'min_pixels_thresh' : 500,
    'tau' : 15.,
    
}


# ## Region Refinement Network parameters

# In[ ]:


rrn_config = {
    
    # Sizes
    'feature_dim' : 64, # 32 would be normal
    'img_H' : 224,
    'img_W' : 224,
    
    # architecture parameters
    'use_coordconv' : False,
    
}


# # UOIS-Net-3D Parameters

# In[ ]:


uois3d_config = {
    
    # Padding for RGB Refinement Network
    'padding_percentage' : 0.25,
    
    # Open/Close Morphology for IMP (Initial Mask Processing) module
    'use_open_close_morphology' : True,
    'open_close_morphology_ksize' : 9,
    
    # Largest Connected Component for IMP module
    'use_largest_connected_component' : True,
    
}


# In[ ]:


checkpoint_dir = 'uois/checkpoints/' # TODO: change this to directory of downloaded models
dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                     dsn_filename,
                                     dsn_config,
                                     rrn_filename,
                                     rrn_config
                                    )


# ## Run on example OSD/OCID images
# 
# We provide a few [OSD](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/osd/) and [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/) images and run the network on them. Evaluation metrics are shown for each of the images.

# In[ ]:


example_images_dir = Path.cwd() / "uois" / "example_images"

img_file_names = img_file_names = sorted(p.name for p in example_images_dir.iterdir())
npy_files = [f for f in img_file_names if f.endswith(".npy")]
print("IMG FILE NAMES: ", npy_files)



#OSD_image_files = sorted(glob.glob(example_images_dir + '/OSD_*.npy'))
#OCID_image_files = sorted(glob.glob(example_images_dir + '/OCID_*.npy'))
N = len(img_file_names) #len(OSD_image_files) + len(OCID_image_files)

rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
label_imgs = np.zeros((N, 480, 640), dtype=np.uint8)

for i, img_file in enumerate(npy_files):#OSD_image_files + OCID_image_files):

    print("---> FILE: ", img_file)
    path = os.path.join(example_images_dir, img_file)
    d = np.load(path, allow_pickle=True, encoding='bytes').item()

    
    
    print(f"====== {img_file} ======")
    print("TYPE: ", type(d))
    print("KEYS: ", d.keys())
    if 'K' in d.keys():
        print("K: ", d['K'])
    if 'seg' in d.keys():
        print("SEG: ", d['seg'])
        print("SHAPE OF SEG: ", d['seg'].shape)
        print("INIQUE VALUES OF d['seg']: ", np.unique(d['seg']))
    if 'label' in d.keys():
        print("LABEL: ", d['label'])
        print("LENGTH LABEL: ", len(d['label']))
        print("TYPE LABEL: ", type(d['label']))
        print("INIQUE VALUES OF d['label']: ", np.unique(d['label']))

   
    if set(d.keys()) == {'rgb', 'depth', 'K', 'seg'}:
        d = cu.convert_cgn_npy_to_uois(d, depth_scale=1.0, out_size=(640, 480))
       
    # RGB
    rgb_img = d['rgb']
    rgb_imgs[i] = data_augmentation.standardize_image(rgb_img)

    # XYZ
    xyz_imgs[i] = d['xyz']

    # Label
    label_imgs[i] = d['label']
    
batch = {
    'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
    'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
}


# In[ ]:
print("Number of images: {0}".format(N))

### Compute segmentation masks ###
st_time = time()
fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
total_time = time() - st_time
print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

# Get results in numpy
seg_masks = seg_masks.cpu().numpy()
fg_masks = fg_masks.cpu().numpy() 
center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
initial_masks = initial_masks.cpu().numpy()


# In[ ]:


rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
total_subplots = 6

fig_index = 1
img_batch = {}
out_cgn_dir = Path("contact_graspnet/test_data_npy_from_uois")
out_cgn_dir.mkdir(parents=True, exist_ok=True)

# set intrinsics (use real ones if you have them)
H, W = rgb_imgs.shape[1], rgb_imgs.shape[2]
fx, fy = 525.0, 525.0
cx, cy = W / 2.0, H / 2.0

for i, name in enumerate(npy_files):
    # --- existing code ---
    num_objs = max(np.unique(seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1

    rgb = rgb_imgs[i].astype(np.uint8)
    xyz = xyz_imgs[i].astype(np.float32)            # (H,W,3)
    depth = xyz[..., 2]

    seg_mask_plot = util_.get_color_mask(seg_masks[i,...], nc=num_objs)
    gt_masks = util_.get_color_mask(label_imgs[i,...], nc=num_objs)

    images = [rgb, depth, seg_mask_plot, gt_masks]
    img_batch[name] = images

    eval_metrics = evaluation.multilabel_metrics(seg_masks[i,...], label_imgs[i])
    print(f"Image {i+1} Metrics:")
    print(eval_metrics)

    # --- NEW: export Contact-GraspNet-format npy ---
    stem = Path(name).stem  # safe for "xxx.npy"
    out_npy = str(out_cgn_dir / f"{stem}.npy")

    cu.uois_to_contactgraspnet(
        rgb=rgb,
        xyz=xyz,
        seg=seg_masks[i, ...],
        out_npy=out_npy,
        fx=fx, fy=fy, cx=cx, cy=cy
    )


# In[ ]:
cu.save_imgs(img_batch, "output/uois")




