#!/usr/bin/env python
# coding: utf-8

# In[1]:


def is_jupyter():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


# # Pip Install

# In[2]:


if is_jupyter():
    get_ipython().system('pip install boto3 astropy sfdmap progressbar2 GPUtil')


# # Initialization

# In[3]:


# Make sure a GPU is available
import GPUtil
print('GPUs:\n{0}'.format('\n'.join(['('+str(i+1)+')\t'+gpu.name for i,gpu in enumerate(GPUtil.getGPUs())])))
import tensorflow as tf
assert tf.config.list_physical_devices('GPU')[0].device_type == 'GPU', 'GPU is not available!'


# In[4]:


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
from s3 import log_s3, s3_save_TF_model
from NN import DistanceLayer, SiameseModel, DistillationDataGenerator


# # Prepare Data

# In[5]:


model_name = 'NN'


# In[6]:


# prepare model paths
s3_model_dir_path = os.path.join(s3_models_dir_path, model_name)
s3_model_train_dir_path = os.path.join(s3_model_dir_path, 'train')
s3_model_test_dir_path = os.path.join(s3_model_dir_path, 'test')
# prepare data paths
s3_data_model_dir_path = os.path.join(s3_data_dir_path, model_name)
s3_data_train_dir_path = os.path.join(s3_data_model_dir_path, 'train')
s3_data_test_dir_path = os.path.join(s3_data_model_dir_path, 'test')


# In[7]:


# load data
dist_mat_path = os.path.join(s3_models_dir_path, 'SmallRF', 'train', 'dist_mat.npy')
dist_mat = from_s3_npy(s3_client, bucket_name, dist_mat_path)
X_train = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_train_dir_path, 'spec.npy'))
X_test = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_test_dir_path, 'spec.npy'))


# In[8]:


# gs_SmallRF_train = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_dir_path, 'SmallRF', 'train', 'gs.pkl')) # <- This is equal to g_NN
gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_model_dir_path, 'gs.pkl'))
gs_train = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_train_dir_path, 'gs.pkl'))
gs_test = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_test_dir_path, 'gs.pkl'))


# In[9]:


I_train = np.array([np.where(gs.index == i)[0][0] for i in gs_train.index])
I_test = np.array([np.where(gs.index == i)[0][0] for i in gs_test.index])


# In[10]:


dist_mat_train = dist_mat[I_train,:][:,I_train]
dist_mat_test = dist_mat[I_test,:][:,I_test]


# In[11]:


if is_jupyter():
    X_train = X_train[:100,:]
    X_test = X_test[:10,:]
    dist_mat_train = dist_mat_train[:100,:][:,:100]
    dist_mat_test = dist_mat_test[:10,:][:,:10]


# # Creating the model

# In[12]:


N_features = X_train.shape[1]


# In[13]:


from tensorflow.keras import applications
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras import initializers

tf.random.set_seed(seed)


# ## Embedding Network

# In[14]:


hidden_size = 512
encoding_size = 128

# input layer
x_in = layers.Input(shape=(N_features, 1))

# adding the network layers
x = x_in
x = layers.Conv1D(64, 31, activation=None, padding='same', kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
x = layers.Conv1D(32, 31, activation=None, padding='same', kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
x = layers.Conv1D(16, 31, activation=None, padding='same', kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
x = layers.Conv1D(8, 31, activation=None, padding='same', kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
x = layers.Conv1D(4, 31, activation=None, padding='same', kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.AveragePooling1D( 2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(hidden_size, kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.relu(x)
x = layers.Dense(encoding_size, kernel_initializer=initializers.GlorotUniform(seed=seed))(x)
x = layers.BatchNormalization()(x)
x = activations.tanh(x)
x_out = x

# creating the model
encoding = Model(x_in, x_out)
encoding.summary()


# ## Siamese Network

# In[15]:


first_input = layers.Input(name="first_input", shape=(N_features))
second_input = layers.Input(name="second_input", shape=(N_features))

first_encoding = encoding(first_input)
second_encoding = encoding(second_input)

distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(first_encoding - second_encoding), -1),1e-9))

siamese_network = Model(
    inputs=[first_input, second_input], outputs=distance
)
siamese_network.summary()


# ## Siamese Model

# In[16]:


siamese_model = SiameseModel(siamese_network, dist_loss='L1')
siamese_model.compile(optimizer=optimizers.Adam(0.001))


# # Train Model

# In[17]:


train_gen = DistillationDataGenerator(X_train, dist_mat_train, batch_size=128, shuffle=True, seed=seed, snr_range_db=[6,40], full_epoch=False, norm=True)
test_gen = DistillationDataGenerator(X_test, dist_mat_test, batch_size=128, shuffle=True, seed=seed, snr_range_db=[6,40], full_epoch=False, norm=True)


# In[18]:


def plot_loss(fig, ax, e, loss_history, val_loss_history):
    if ax.lines:
        for i,line in enumerate(ax.lines):
            line.set_xdata(e)
            if (i==1):
                line.set_ydata(loss_history)
            else:
                line.set_ydata(val_loss_history)
    else:
        ax.plot(e, loss_history, label='training')
        ax.plot(e, val_loss_history, label='test')
        ax.legend()
    fig.canvas.draw()


# In[19]:


epochs = 50
sub_epochs = 5
N_chunks = int(epochs/sub_epochs)
loss_history = []
val_loss_history = []
verbosity = 1 if is_jupyter() else 2

# training loop
print('Training for {0} full epochs, and stopping for saving every {1} full epochs, for a total of {2} stages.'.format(epochs,sub_epochs, N_chunks))
start_time = time.time()
for i_chunk in range(N_chunks):
    
    print('-------------------------------------')
    print('epochs {0}-{1}:'.format(i_chunk*sub_epochs+1, (i_chunk+1)*sub_epochs))
    print('-------------------------------------')

    # train
    try:
        # for some reason, the first call to fit will throw KeyError...
        history = siamese_model.fit(train_gen, epochs=sub_epochs, validation_data=test_gen, verbose=verbosity)
    except KeyError:
        history = siamese_model.fit(train_gen, epochs=sub_epochs, validation_data=test_gen, verbose=verbosity)
    loss_history += history.history['loss']
    val_loss_history += history.history['val_loss']
    
    # create the figures for the loss
    loss_fig, loss_ax = plt.subplots(figsize=(15,8))
    loss_ax.set_title('Training curve')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.grid()
    log_loss_fig, log_loss_ax = plt.subplots(figsize=(15,8))
    log_loss_ax.set_title('Training curve (Log Scale)')
    log_loss_ax.set_xlabel('epoch')
    log_loss_ax.set_ylabel('log(loss)')
    log_loss_ax.grid()
    log_loss_ax.set_yscale('log')
    
    # plot the loss
    curr_epochs = (i_chunk+1)*sub_epochs
    e = np.arange(curr_epochs)+1
    """
    loss_ax.plot(e, loss_history, label='training')
    loss_ax.plot(e, val_loss_history, label='test')
    loss_ax.legend()
    log_loss_ax.plot(e, loss_history, label='training')
    log_loss_ax.plot(e, val_loss_history, label='test')
    log_loss_ax.legend()
    """
    plot_loss(loss_fig, loss_ax, e, loss_history, val_loss_history)
    plot_loss(log_loss_fig, log_loss_ax, e, loss_history, val_loss_history)
    plt.show()
    time.sleep(1)
    
    end_time = time.time()
    time_str = 'TOTAL TIME = {0:.3f} hours'.format((end_time - start_time)/3600)
    print(time_str)
    
    # create a sub dir
    s3_save_NN_dir_path_sub_epoch = os.path.join(s3_model_train_dir_path, 'after_{0}_epochs'.format((i_chunk+1)*sub_epochs))
    # save the figures
    to_s3_fig(loss_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.png'))
    to_s3_fig(log_loss_fig, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.png'))
    # save the losses
    to_s3_npy(np.array(loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'loss.npy'))
    to_s3_npy(np.array(val_loss_history), s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'val_loss.npy'))
    # get model summary
    stringlist = []
    encoding.summary(print_fn=lambda x: stringlist.append(x))
    encoding_summary = "\n".join(stringlist)
    stringlist = []
    siamese_network.summary(print_fn=lambda x: stringlist.append(x))
    siamese_network_summary = "\n".join(stringlist)
    # save log
    log_s3(s3_client, bucket_name, s3_model_train_dir_path, 'NN_log.txt',
        dist_mat_path = dist_mat_path,
        s3_model_train_dir_path = s3_model_train_dir_path,
        training_duration = time_str,
        encoding_summary = encoding_summary,
        siamese_network_summary = siamese_network_summary
        )
    # save the network
    s3_model_path = os.path.join(s3_save_NN_dir_path_sub_epoch, 'model')
    s3_save_TF_model(siamese_model, s3_client, bucket_name, s3_model_path)


# # Inference

# In[20]:


# predict
batch_size = 128
data_gen = DistillationDataGenerator(X_train,  dist_mat_train, batch_size=128, shuffle=False, seed=seed, full_epoch=True, norm=True)
Z_NN = siamese_model.predict(data_gen, verbose=verbosity)


# In[21]:


# create full distance matrix
N = int((-1+np.sqrt(1+8*len(Z_NN)))/2)
D_NN = np.zeros(shape=(N,N))
D_NN[np.triu_indices(N)] = Z_NN
D_NN = D_NN.T
D_NN[np.triu_indices(N)] = Z_NN


# In[22]:


# save the distance matrix
to_s3_npy(D_NN, s3_client, bucket_name, os.path.join(s3_save_NN_dir_path_sub_epoch, 'dist_mat.npy'))

