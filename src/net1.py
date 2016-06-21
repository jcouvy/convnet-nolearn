# -------------------- Loading Data-set --------------------

import cPickle
import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read training and test data files
train = pd.read_csv("input/train.csv")
test  = pd.read_csv("input/test.csv")

train_images = train.iloc[:,1:].values
train_labels = train[[0]].values.ravel()

# Reshape and normalize training data
trainX = train_images.reshape(train.shape[0], 1, 28, 28).astype(np.float32)
trainX /= 255.0

# Reshape and normalize test data
testX = test.values.reshape(test.shape[0], 1, 28, 28).astype(np.float32)
testX /= 255.0

trainY = train_labels.astype(np.int32)

# -------------------- Building the network -------------------

import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, PrintLayerInfo #used for verbose=2

net1 = NeuralNet(
    layers = [('input', layers.InputLayer),
              ('conv1', layers.Conv2DLayer),
              ('pool1', layers.MaxPool2DLayer),
              ('conv2', layers.Conv2DLayer),
              ('pool2', layers.MaxPool2DLayer),
              ('hidden', layers.DenseLayer),
              ('output', layers.DenseLayer)
              ],
    
    # Each digit is a 28x28 1-Dimensionnal image,
    input_shape = (None, 1, 28, 28),

    conv1_num_filters =  8, conv1_filter_size = (3, 3),
    pool1_pool_size = (2, 2),

    conv2_num_filters = 16, conv2_filter_size = (2, 2),
    pool2_pool_size = (2, 2),

    hidden_num_units = 64,
    output_num_units = 10,

    # Learning method.
    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,

    # Length of the training
    max_epochs = 100,
    verbose = 2, 

)

# Start training the network
net1.fit(trainX, trainY)

# Saving the net with cPickle to load it back later.
with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)

