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

# Converts numbers into 32b floats best used w/ GPUs.
def float32(k):
    return np.cast['float32'](k)

# Allows fine tuning of hyper-parameters during the learning process,
# Credits to (@Karpathy)
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start = start
        self.stop = stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

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


net3 = NeuralNet(
    layers = [('input', layers.InputLayer),
              ('conv1', layers.Conv2DLayer),
              ('pool1', layers.MaxPool2DLayer),
              ('conv2', layers.Conv2DLayer),
              ('pool2', layers.MaxPool2DLayer),
              ('hidden1', layers.DenseLayer),
              ('dropout', layers.DropoutLayer),
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

    dropout_p = 0.3,
    hidden1_num_units = 128, hidden2_num_units = 128,
    output_num_units = 10,
    output_nonlinearity = lasagne.nonlinearities.softmax, #Softmax classifier (default value)

    # Calls the class CropBatchIterator while training occurs
    # Doesn't affect GPU performance as the CPU deals with the
    # cropping operation.
    batch_iterator_train = CropBatchIterator(batch_size=200),
    batch_iterator_test = CropBatchIterator(batch_size=200),

    # Learning method.
    update = nesterov_momentum,
    # The parameters have to be converted to theano shared vars in order
    # to be changed later on.
    update_learning_rate = theano.shared(float32(0.03)),
    update_momentum = theano.shared(float32(0.9)),    

    # NeuralNet allows us to dynamically tune its parameters. We use the class
    # AdjustVariable to decrease the learning rate with the number of epochs
    # while the momentum increases.
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],

    regression = False,
    max_epochs = 1000,

    # Deeper layer information (learning capacity, coverage...)
    verbose = 2, 

)

# Start training the network.
net3.fit(trainX, trainY)

# Raise the the recursion limit to prevent a cPickle error.
sys.setrecursionlimit(10000)
# # Saving the net with cPickle to load it back later.
with open('net3_github.pickle', 'wb') as f:
   cPickle.dump(net3, f, -1)

