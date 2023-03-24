#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
    __IPYTHON__
    is_notebook = True
    print('Notebook mode')
except NameError:
    is_notebook = False
    print('Script mode')


# In[ ]:


if is_notebook:
    i_slice = 0
    n_slices = 16
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="the name of the model", type=str)
    parser.add_argument("i_slice", help="slice index", type=int)
    parser.add_argument("n_slices", help="number of slices", type=int)
    args = parser.parse_args()
    print('args:\n\t'+'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))
    MODEL_NAME = args.model_name
    i_slice = args.i_slice
    n_slices = args.n_slices


# # Pip Install

# In[ ]:


if is_notebook:
    get_ipython().system('pip install boto3 astropy sfdmap progressbar2 GPUtil parse')


# # Initialization

# In[ ]:


import GPUtil
print('GPUs:\n{0}'.format('\n'.join(['('+str(i+1)+')\t'+gpu.name+'\t-\t{:.2f}GB'.format(gpu.memoryFree/1e3) for i,gpu in enumerate(GPUtil.getGPUs())])))
N_GPUs = len(GPUtil.getGPUs())

# setting environ variables using !
if is_notebook:
    get_ipython().system('export TF_GPU_THREAD_MODE="gpu_private"')
    if N_GPUs>1:
        if N_GPUs==2:
            get_ipython().system('export TF_MIN_GPU_MULTIPROCESSOR_COUNT=2')
            get_ipython().system('export CUDA_VISIBLE_DEVICES="0,1"')
        if N_GPUs==3:
            get_ipython().system('export TF_MIN_GPU_MULTIPROCESSOR_COUNT=3')
            get_ipython().system('export CUDA_VISIBLE_DEVICES="0,1,2"')
        if N_GPUs==4:
            get_ipython().system('export TF_MIN_GPU_MULTIPROCESSOR_COUNT=4')
            get_ipython().system('export CUDA_VISIBLE_DEVICES="0,1,2,3"')
        if N_GPUs==5:
            get_ipython().system('export TF_MIN_GPU_MULTIPROCESSOR_COUNT=5')
            get_ipython().system('export CUDA_VISIBLE_DEVICES="0,1,2,3,4"')
        if N_GPUs==6:
            get_ipython().system('export TF_MIN_GPU_MULTIPROCESSOR_COUNT=6')
            get_ipython().system('export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"')
        if N_GPUs==7:
            get_ipython().system('export TF_MIN_GPU_MULTIPROCESSOR_COUNT=7')
            get_ipython().system('export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"')
        if N_GPUs==8:
            get_ipython().system('export TF_MIN_GPU_MULTIPROCESSOR_COUNT=8')
            get_ipython().system('export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"')

# set environ variables using os.environ
import os 
if N_GPUs>1:
    os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]=str(N_GPUs)
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in range(N_GPUs)])
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"

# make sure tensorflow detect the GPUs
import tensorflow as tf
assert len(tf.config.list_physical_devices('GPU'))==N_GPUs, 'Not all GPUs are available!'


# In[ ]:


# imports
import pandas as pd
import numpy as np
import os
import sys
import pickle
import boto3
from matplotlib import pyplot as plt
import time

# random seed
seed = 42
np.random.seed(seed)

# local files paths
local_home_dir_path = os.path.expanduser("~")
local_work_dir_path = os.path.join(local_home_dir_path, 'thesis2')
local_code_dir_path = os.path.join(local_work_dir_path , 'code')

# S3 file paths
endpoint_url = 'https://s3-west.nrp-nautilus.io'
bucket_name = 'tau-astro'
prefix = 'almogh'
s3_work_dir_path = os.path.join(prefix, 'thesis2')
s3_data_dir_path = os.path.join(s3_work_dir_path , 'data')
s3_models_dir_path = os.path.join(s3_work_dir_path , 'models')
s3_final_table_csv_path = os.path.join(s3_data_dir_path, 'SDSS_DR16_all.csv')

s3_client = boto3.client("s3", endpoint_url=endpoint_url)

# adding code folder to path
sys.path.insert(1, local_code_dir_path)
from s3 import to_s3_npy, to_s3_pkl, from_s3_npy, from_s3_pkl, to_s3_fig
from s3 import log_s3, s3_save_TF_model, s3_load_TF_model
from NN import DistanceLayer, SiameseModel, DistillationDataGenerator, L1, L2


# # Load Data

# In[ ]:


data_path_in_bucket = 'almogh/thesis2/data/BigRF/train/spec.npy'
X = from_s3_npy(s3_client = s3_client,
                bucket_name = bucket_name,
                path_in_bucket = data_path_in_bucket)


# # Load the model and infer

# In[ ]:


from NN import DistanceLayer, L1, L2, SiameseModel, DistillationDataGenerator, full_dist_mat_from_upper_diag_part
from s3 import s3_download_model
import tempfile
from tensorflow.keras import optimizers, mixed_precision

custom_objects = {'DistanceLayer': DistanceLayer, 'L1':L1, 'L2':L2, 'SiameseModel':SiameseModel}
batch_size = 128*N_GPUs
verbosity = 2

if N_GPUs>1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

with strategy.scope():

    # load model from S3
    model_name = 'model'
    with tempfile.TemporaryDirectory() as tempdir:
        s3_download_model(s3_client = s3_client,
                          bucket_name = bucket_name,
                          path_in_bucket = 'almogh/thesis2/models/{0}/train/after_50_epochs/model'.format(MODEL_NAME),
                          model_name = model_name,
                          tempdir = tempdir)
        NN = tf.keras.models.load_model(f"{tempdir}/{model_name}", custom_objects=custom_objects)
        
    # compile the model
    optimizer = optimizers.Adam(0.001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    NN.compile(optimizer=optimizer)
    
    # predict
    data_gen = DistillationDataGenerator(X, D=None, batch_size=batch_size, shuffle=False, seed=42, full_epoch=True, norm=True, i_slice=i_slice, n_slices=n_slices)
    Z_NN = NN.predict(data_gen, verbose=verbosity, workers=2*N_GPUs, use_multiprocessing=True)


# In[ ]:


# save
to_s3_npy(Z_NN,
          s3_client = s3_client,
          bucket_name = bucket_name,
          path_in_bucket = 'almogh/thesis2/eval/inference/Z_{0}_i{1}_n{2}.npy'.format(MODEL_NAME, i_slice, n_slices))

