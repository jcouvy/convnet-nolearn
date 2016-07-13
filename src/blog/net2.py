# -------------------- Loading Data-set --------------------

import cPickle
import pandas as pd
import numpy as np
import sys

# The competition datafiles are in the directory ../input
# Read training and test data files.
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
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo

# Custom Batch Iterator called by NeuralNet to provided augmented
# data in order to reduce overfitting.
# Randomly crops by 2 pixels the inputed image.
# Returns the new tensor Xb (trainX) and the labels.
class CropBatchIterator(BatchIterator):
    cropX, cropY = 2, 2
    def transform(self, Xb, yb):
        Xb, yb = super(CropBatchIterator, self).transform(Xb, yb)
        shape = Xb.shape[0]
        width = 28 - self.cropX
        height = 28 - self.cropY
        new_Xb = np.zeros([shape, 1, width, height], dtype=np.float32)
        for i in range (shape):
            dx = np.random.randint(self.cropX+1)
            dy = np.random.randint(self.cropY+1)
            new_Xb[i] = Xb[i, :, dy:dy+width, dx:dx+height]
        return new_Xb, yb


net2 = NeuralNet(
    layers = [('input', layers.InputLayer),
              ('conv1', layers.Conv2DLayer),
              ('pool1', layers.MaxPool2DLayer),
              ('conv2', layers.Conv2DLayer),
              ('pool2', layers.MaxPool2DLayer),
              ('hidden1', layers.DenseLayer),
              ('hidden2', layers.DenseLayer),
              ('output', layers.DenseLayer)
              ],
    # Each digit is a 28x28 1-Dimensionnal image,
    # We crop each image by 2 pixels in width and height.
    input_shape = (None, 1, 26, 26),

    conv1_num_filters =  13, conv1_filter_size = (3, 3),
    conv1_nonlinearity = lasagne.nonlinearities.rectify, #ReLu rectifier (default value)
    pool1_pool_size = (2, 2),
    
    conv2_num_filters = 26, conv2_filter_size = (2, 2),
    conv2_nonlinearity = lasagne.nonlinearities.rectify,
    pool2_pool_size = (2, 2),

    hidden1_num_units = 128, hidden2_num_units = 128,
    output_num_units = 10,  output_nonlinearity = lasagne.nonlinearities.softmax, #Softmax classifier (default value)

    # Learning method.
    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,    

    # Calls the class CropBatchIterator while training occurs
    # Doesn't affect GPU performance as the CPU deals with the
    # cropping operation.
    batch_iterator_train = CropBatchIterator(batch_size=200),
    batch_iterator_test = CropBatchIterator(batch_size=200),

    regression = False,
    max_epochs = 500,

    # Deeper layer information (learning capacity, coverage...)
    verbose = 2, 

)

# Start training the network.
net2.fit(trainX, trainY)

# Raise the the recursion limit to prevent a cPickle error.
sys.setrecursionlimit(10000)
# Saving the net with cPickle to load it back later.
with open('net2.pickle', 'wb') as f:
   pickle.dump(net2, f, -1)

