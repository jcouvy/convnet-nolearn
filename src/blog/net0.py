# -------------------- Loading Data-set --------------------

import cPickle
import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read training and test data files
train = pd.read_csv("../../input/train.csv")
test  = pd.read_csv("../../input/test.csv")

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
from lasagne.updates import sgd
from nolearn.lasagne import NeuralNet

net0 = NeuralNet(
    layers = [('input', layers.InputLayer),
              ('conv1', layers.Conv2DLayer),
              ('conv2', layers.Conv2DLayer),
              ('output', layers.DenseLayer)
              ],
    
    # Each digit is a 28x28 1-Dimensionnal image,
    input_shape = (None, 1, 28, 28),

    conv1_num_filters =  8, conv1_filter_size = (3, 3),
    conv1_nonlinearity = lasagne.nonlinearities.rectify,
    
    conv2_num_filters = 16, conv2_filter_size = (2, 2),
    conv2_nonlinearity = lasagne.nonlinearities.rectify,  # Default value

    output_num_units = 10,
    output_nonlinearity = lasagne.nonlinearities.softmax, # Default Value

    # Learning method.
    update = sgd,
    update_learning_rate = 0.01,

    # Length of the training
    max_epochs = 100,
    verbose = 1, 

)

# Start training the network
net0.fit(trainX, trainY)

# Saving the net with cPickle to load it back later.
 with open('net0.pickle', 'wb') as f:
    pickle.dump(net0, f, -1)

