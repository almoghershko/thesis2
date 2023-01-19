#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    __IPYTHON__
    is_notebook = True
    print('Notebook mode')
except NameError:
    is_notebook = False
    print('Script mode')


# In[2]:


IsSmallDataSlice = True
IsShortEpochs = False
IsShortTraining = True
IsSaveModel = True


# # Pip Install

# In[3]:


if is_notebook:
    get_ipython().system('pip install boto3 astropy sfdmap progressbar2 GPUtil')


# # Initialization

# In[4]:


import GPUtil
print('GPUs:\n{0}'.format('\n'.join(['('+str(i+1)+')\t'+gpu.name for i,gpu in enumerate(GPUtil.getGPUs())])))


# In[5]:


N_GPUs = len(GPUtil.getGPUs())


# In[6]:


gpu = GPUtil.getGPUs()[0]


# In[7]:


gpu.memoryFree


# In[8]:


get_ipython().system('echo TF_GPU_THREAD_MODE="gpu_private"')
if N_GPUs>1:
    if N_GPUs==2:
        get_ipython().system('echo TF_MIN_GPU_MULTIPROCESSOR_COUNT=2')
        get_ipython().system('echo CUDA_VISIBLE_DEVICES="0,1"')
    if N_GPUs==3:
        get_ipython().system('echo TF_MIN_GPU_MULTIPROCESSOR_COUNT=3')
        get_ipython().system('echo CUDA_VISIBLE_DEVICES="0,1,2"')
    if N_GPUs==4:
        get_ipython().system('echo TF_MIN_GPU_MULTIPROCESSOR_COUNT=4')
        get_ipython().system('echo CUDA_VISIBLE_DEVICES="0,1,2,3"')
    if N_GPUs==5:
        get_ipython().system('echo TF_MIN_GPU_MULTIPROCESSOR_COUNT=5')
        get_ipython().system('echo CUDA_VISIBLE_DEVICES="0,1,2,3,4"')
    if N_GPUs==6:
        get_ipython().system('echo TF_MIN_GPU_MULTIPROCESSOR_COUNT=6')
        get_ipython().system('echo CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"')
    if N_GPUs==7:
        get_ipython().system('echo TF_MIN_GPU_MULTIPROCESSOR_COUNT=7')
        get_ipython().system('echo CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"')
    if N_GPUs==8:
        get_ipython().system('echo TF_MIN_GPU_MULTIPROCESSOR_COUNT=8')
        get_ipython().system('echo CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"')


# In[9]:


import os 
if N_GPUs>1:
    os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]=str(N_GPUs)
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in range(N_GPUs)])
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"


# In[10]:


import tensorflow as tf
assert len(tf.config.list_physical_devices('GPU'))==N_GPUs, 'Not all GPUs are available!'


# In[11]:


tf.config.list_physical_devices('GPU')


# In[12]:


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

# In[13]:


data_model_name = 'NN'
save_model_name = 'NN2'


# In[14]:


# prepare model paths
s3_model_dir_path = os.path.join(s3_models_dir_path, save_model_name)
s3_model_train_dir_path = os.path.join(s3_model_dir_path, 'train')
s3_model_test_dir_path = os.path.join(s3_model_dir_path, 'test')
# prepare data paths
s3_data_model_dir_path = os.path.join(s3_data_dir_path, data_model_name)
s3_data_train_dir_path = os.path.join(s3_data_model_dir_path, 'train')
s3_data_val_dir_path = os.path.join(s3_data_model_dir_path, 'val')
s3_data_test_dir_path = os.path.join(s3_data_dir_path, 'SmallRF', 'test')


# In[15]:


# load data
dist_mat_path = os.path.join(s3_models_dir_path, 'SmallRF', 'train', 'dist_mat.npy')
dist_mat = from_s3_npy(s3_client, bucket_name, dist_mat_path)
X_train = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_train_dir_path, 'spec.npy'))
X_val = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_val_dir_path, 'spec.npy'))
X_test = from_s3_npy(s3_client, bucket_name, os.path.join(s3_data_test_dir_path, 'spec.npy'))


# In[16]:


# gs_SmallRF_train = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_dir_path, 'SmallRF', 'train', 'gs.pkl')) # <- This is equal to g_NN
gs = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_model_dir_path, 'gs.pkl'))
gs_train = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_train_dir_path, 'gs.pkl'))
gs_val = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_val_dir_path, 'gs.pkl'))
gs_test = from_s3_pkl(s3_client, bucket_name, os.path.join(s3_data_test_dir_path, 'gs.pkl'))


# In[17]:


I_train = np.array([np.where(gs.index == i)[0][0] for i in gs_train.index])
I_val = np.array([np.where(gs.index == i)[0][0] for i in gs_val.index])


# In[18]:


dist_mat_train = dist_mat[I_train,:][:,I_train]
dist_mat_val = dist_mat[I_val,:][:,I_val]


# In[19]:


if IsSmallDataSlice:
    print('Running on a tiny slice of the data')
    N_nb = 500
    X_train = X_train[:N_nb,:]
    X_val = X_val[:N_nb,:]
    X_test = X_test[:N_nb,:]
    dist_mat_train = dist_mat_train[:N_nb,:][:,:N_nb]
    dist_mat_val = dist_mat_val[:N_nb,:][:,:N_nb]
    gs_train = gs_train[:N_nb]
    gs_val = gs_val[:N_nb]
    gs_test = gs_test[:N_nb]


# # Creating the model

# In[20]:


N_features = X_train.shape[1]


# In[21]:


from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# In[22]:


from tensorflow.keras import applications
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras import initializers

from NN import DistanceLayer

tf.random.set_seed(seed)


# In[23]:


hidden_size = 512
encoding_size = 128


# In[24]:


if N_GPUs>1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()


# In[25]:


with strategy.scope():

    ##############################
    #     Embedding Network      #
    ##############################
    
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
    #x = activations.tanh(x, dtype='float32')
    x = layers.Activation('tanh', dtype='float32', name='encoding')(x)
    x_out = x

    # creating the model
    encoding = Model(x_in, x_out)
    
    ##############################
    #     Siamese Network        #
    ##############################
    first_input = layers.Input(name="first_input", shape=(N_features))
    second_input = layers.Input(name="second_input", shape=(N_features))

    first_encoding = encoding(first_input)
    second_encoding = encoding(second_input)

    #distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(first_encoding - second_encoding), -1),1e-9))
    distance = DistanceLayer(dtype=mixed_precision.Policy('float32'))(first_encoding, second_encoding)

    siamese_network = Model(
        inputs=[first_input, second_input], outputs=distance
    )
    
    ##############################
    #     Siamese Model          #
    ##############################
    
    siamese_model = SiameseModel(siamese_network, dist_loss='L1')

optimizer = optimizers.Adam(0.001)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
siamese_model.compile(optimizer=optimizer)


# In[26]:


encoding.summary()


# In[27]:


siamese_network.summary()


# # Train Model

# In[28]:


if IsShortEpochs:
    print('Running short epochs')
    full_epoch = False
else:
    print('Running full epochs')
    full_epoch = True
    
batch_size = 128*N_GPUs

train_gen = DistillationDataGenerator(X_train, dist_mat_train, batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=[6,40], full_epoch=full_epoch, norm=True)
val_gen = DistillationDataGenerator(X_val, dist_mat_val, batch_size=batch_size, shuffle=True, seed=seed, snr_range_db=[6,40], full_epoch=full_epoch, norm=True)


# In[29]:


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
        ax.plot(e, val_loss_history, label='validation')
        ax.legend()
    fig.canvas.draw()


# In[30]:


# Create a TensorBoard callback
from datetime import datetime
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
profile_batch = '{0},{1}'.format(str(int(len(train_gen)/2)),str(20+int(len(train_gen)/2)))
print('profile_batch={0}'.format(profile_batch))
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = profile_batch)


# In[31]:


if IsShortTraining:
    epochs = 1
    sub_epochs = 1
else:
    epochs = 50
    sub_epochs = 5
N_chunks = int(epochs/sub_epochs)
loss_history = []
val_loss_history = []
verbosity = 1 if is_notebook else 2

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
        history = siamese_model.fit(train_gen, epochs=sub_epochs, validation_data=val_gen, verbose=verbosity, callbacks = [tboard_callback], workers=N_GPUs, use_multiprocessing=True)
        #history = siamese_model.fit(train_gen, epochs=sub_epochs, validation_data=val_gen, verbose=verbosity, callbacks = [tboard_callback])
    except KeyError:
        history = siamese_model.fit(train_gen, epochs=sub_epochs, validation_data=val_gen, verbose=verbosity, callbacks = [tboard_callback], workers=N_GPUs, use_multiprocessing=True)
        #history = siamese_model.fit(train_gen, epochs=sub_epochs, validation_data=val_gen, verbose=verbosity, callbacks = [tboard_callback])
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
    plot_loss(loss_fig, loss_ax, e, loss_history, val_loss_history)
    plot_loss(log_loss_fig, log_loss_ax, e, loss_history, val_loss_history)
    plt.show()
    time.sleep(1)
    
    end_time = time.time()
    time_str = 'TOTAL TIME = {0:.3f} hours'.format((end_time - start_time)/3600)
    print(time_str)
    
    if IsSaveModel:
    
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

# In[ ]:


def infer_dist_mat(model, X, verbosity):
    # predict
    data_gen = DistillationDataGenerator(X, np.zeros(shape=(X.shape[0], X.shape[0])), batch_size=128, shuffle=False, seed=seed, full_epoch=True, norm=True)
    Z_NN = model.predict(data_gen, verbose=verbosity)
    # create full distance matrix
    N = int((-1+np.sqrt(1+8*len(Z_NN)))/2)
    D_NN = np.zeros(shape=(N,N))
    D_NN[np.triu_indices(N)] = Z_NN
    D_NN = D_NN.T
    D_NN[np.triu_indices(N)] = Z_NN
    return D_NN


# ## Training set

# In[ ]:


dist_mat = infer_dist_mat(siamese_model, X_train, verbosity)
to_s3_npy(dist_mat, s3_client, bucket_name, os.path.join(s3_model_train_dir_path, 'dist_mat.npy'))


# In[ ]:


weird_scores = np.mean(dist_mat, axis=1)
to_s3_npy(weird_scores, s3_client, bucket_name, os.path.join(s3_model_train_dir_path, 'weird_scores.npy'))


# In[ ]:


from sklearn.manifold import TSNE
sne = TSNE(n_components=2, perplexity=25, metric='precomputed', verbose=1, random_state=seed).fit_transform(dist_mat)
to_s3_npy(sne, s3_client, bucket_name, os.path.join(s3_model_train_dir_path, 'tsne.npy'))


# In[ ]:


fig = plt.figure()
tmp = plt.hist(weird_scores, bins=60, color="g")
plt.title("Weirdness score histogram")
plt.ylabel("N")
plt.xlabel("weirdness score")
to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_train_dir_path, 'weirdness_scores_histogram.png'))


# In[ ]:


distances = dist_mat[np.tril_indices(dist_mat.shape[0], -1)]

fig = plt.figure()
tmp = plt.hist(distances, bins=100)
plt.title("Distances histogram")
plt.ylabel("N")
plt.xlabel("distance")

to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_train_dir_path, 'distances_histogram.png'))


# In[ ]:


sne_f1 = sne[:, 0]
sne_f2 = sne[:, 1]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
im_scat = ax.scatter(sne_f1, sne_f2, s=3, c=weird_scores, cmap=plt.cm.get_cmap('jet'), picker=1)
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.set_title(r't-SNE Scatter Plot Colored by Weirdness score')
clb = fig.colorbar(im_scat, ax=ax)
clb.ax.set_ylabel('Weirdness', rotation=270)
plt.show()

to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_train_dir_path, 'tsne_colored_by_weirdness.png'))


# In[ ]:


snr = gs_train.snMedian

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
import matplotlib.colors as colors
im_scat = ax.scatter(sne_f1, sne_f2, s=3, c=snr, cmap=plt.cm.get_cmap('jet'), norm=colors.LogNorm(vmin=snr.min(), vmax=80))
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.set_title(r't-SNE Scatter Plot Colored by SNR')
clb = fig.colorbar(im_scat, ax=ax)
clb.ax.set_ylabel('SNR', rotation=270)
plt.show()

to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_train_dir_path, 'tsne_colored_by_snr.png'))


# ## Test set

# In[ ]:


dist_mat_test = infer_dist_mat(siamese_model, X_test, verbosity)
to_s3_npy(dist_mat_test, s3_client, bucket_name, os.path.join(s3_model_test_dir_path, 'dist_mat.npy'))


# In[ ]:


weird_scores_test = np.mean(dist_mat_test, axis=1)
to_s3_npy(weird_scores_test, s3_client, bucket_name, os.path.join(s3_model_test_dir_path, 'weird_scores.npy'))


# In[ ]:


from sklearn.manifold import TSNE
sne_test = TSNE(n_components=2, perplexity=25, metric='precomputed', verbose=1, random_state=seed).fit_transform(dist_mat_test)
to_s3_npy(sne_test, s3_client, bucket_name, os.path.join(s3_model_test_dir_path, 'tsne.npy'))


# In[ ]:


fig = plt.figure()
tmp = plt.hist(weird_scores_test, bins=60, color="g")
plt.title("Weirdness score histogram")
plt.ylabel("N")
plt.xlabel("weirdness score")
to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_test_dir_path, 'weirdness_scores_histogram.png'))


# In[ ]:


distances_test = dist_mat_test[np.tril_indices(dist_mat_test.shape[0], -1)]

fig = plt.figure()
tmp = plt.hist(distances_test, bins=100)
plt.title("Distances histogram")
plt.ylabel("N")
plt.xlabel("distance")

to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_test_dir_path, 'distances_histogram.png'))


# In[ ]:


sne_f1_test = sne_test[:, 0]
sne_f2_test = sne_test[:, 1]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
im_scat = ax.scatter(sne_f1_test, sne_f2_test, s=3, c=weird_scores_test, cmap=plt.cm.get_cmap('jet'), picker=1)
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.set_title(r't-SNE Scatter Plot Colored by Weirdness score')
clb = fig.colorbar(im_scat, ax=ax)
clb.ax.set_ylabel('Weirdness', rotation=270)
plt.show()

to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_test_dir_path, 'tsne_colored_by_weirdness.png'))


# In[ ]:


snr_test = gs_test.snMedian

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
import matplotlib.colors as colors
im_scat = ax.scatter(sne_f1_test, sne_f2_test, s=3, c=snr_test, cmap=plt.cm.get_cmap('jet'), norm=colors.LogNorm(vmin=snr.min(), vmax=80))
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.set_title(r't-SNE Scatter Plot Colored by SNR')
clb = fig.colorbar(im_scat, ax=ax)
clb.ax.set_ylabel('SNR', rotation=270)
plt.show()

to_s3_fig(fig, s3_client, bucket_name, os.path.join(s3_model_test_dir_path, 'tsne_colored_by_snr.png'))

