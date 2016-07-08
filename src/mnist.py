import theano
import numpy as np

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, rectify
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

    
from nolearn.lasagne import (NeuralNet,
                             BatchIterator,
                             PrintLayerInfo,
                             TrainSplit)
from nolearn.lasagne import visualize


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
    

DATA_PATH = '/net/www/jcouvy/data/'

# -------------------- Loading Data-set --------------------

def load_data():
    # The competition datafiles are in the directory ../input
    # Read training and test data files.
    train = pd.read_csv(DATA_PATH+"train.csv")
    test  = pd.read_csv(DATA_PATH+"test.csv")

    train_images = train.iloc[:,1:].values
    train_labels = train[[0]].values.ravel()

    # Reshape and normalize training data
    X_train = train_images.reshape(train.shape[0], 1, 28, 28).astype(np.float32)
    X_train -= X_train.mean()
    X_train /= X_train.std()

    # Reshape and normalize test data
    X_test = test.values.reshape(test.shape[0], 1, 28, 28).astype(np.float32)
    X_test -= X_train.mean()
    X_test /= X_train.std()

    y_train = train_labels.astype(np.int32)
    
    return X_train, y_train, X_test

def visualize_data(X, y):
    figs, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(-X[i + 4 * j].reshape(28, 28), cmap='gray', interpolation='none')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
            axes[i, j].axis('off')

# --------------- Network architectures ---------------
        
dropout_net = [
    (layers.InputLayer, {'shape': (None, 1, 28, 28)}),

    (layers.Conv2DLayer, {'num_filters': 16, 'filter_size': 3}),
    (layers.MaxPool2DLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.3}),

    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 3}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 3}),
    (layers.MaxPool2DLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.3}),

    (layers.DenseLayer, {'num_units': 100, 'nonlinearity': rectify}),
    (layers.DropoutLayer, {'p': 0.3}),
    (layers.DenseLayer, {'num_units': 10,   'nonlinearity': softmax}),
    ]
    

network_in_network = [
    (layers.InputLayer,  {'shape': (None, 1, 28, 28)}),

    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 5, 'pad': 2, 'stride': 1}),
    (layers.NINLayer, {'num_units': 16}),
    (layers.NINLayer, {'num_units': 16}),
    (layers.DropoutLayer, {'p': 0.5}),

    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 5, 'pad': 2, 'stride': 1}),
    (layers.NINLayer, {'num_units': 16}),
    (layers.NINLayer, {'num_units': 16}),
    (layers.DropoutLayer, {'p': 0.5}),
        
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 5, 'pad': 2, 'stride': 1}),
    (layers.NINLayer, {'num_units': 16}),
    (layers.NINLayer, {'num_units': 10}),

    (layers.GlobalPoolLayer, {}),
    (layers.DenseLayer, {'num_units':  10, 'nonlinearity': softmax})       
    ]

    
deep_convnet = [
    (layers.InputLayer, {'shape': (None, 1, 24, 24)}),

    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
    (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),

    (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
    (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),

    (layers.DenseLayer, {'num_units': 64, 'nonlinearity': rectify}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 64, 'nonlinearity': softmax}),
    ]

# -------------------- Building the network -------------------


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
# Returns the new tensor Xb (X_train) and the labels.
class CropBatchIterator(BatchIterator):
    cropX, cropY = 4, 4
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

    
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        # if yb is not None:
        #     # Horizontal flip of all x coordinates:
        #     yb[indices, ::2] = yb[indices, ::2] * -1

        return Xb, yb

        
def build_network(layers):    
    return NeuralNet(
        layers = layers,

        batch_iterator_train = FlipBatchIterator(batch_size=128),

        update = nesterov_momentum,
        update_learning_rate = theano.shared(float32(0.03)),
        update_momentum = theano.shared(float32(0.9)),    

        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=50),
            ],

        train_split = TrainSplit(eval_size=0.2),

        regression = False,
        max_epochs = 100,
        verbose = 2
        )

# --------------- Training the network ---------------


X_train, y_train, X_test = load_data()
visualize_data(X_train, y_train)

mnist = build_network(network_in_network)
mnist.fit(X_train, y_train)
mnist.predict(X_test)

visualize.plot_loss(mnist)
plt.savefig("../results/mnist/plotloss.png")
