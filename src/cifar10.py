from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, rectify
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo, TrainSplit
from nolearn.lasagne import visualize

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from urllib import urlretrieve

import os
import sys
import tarfile
import cPickle
import random

import theano
import lasagne
import numpy as np
import matplotlib.pyplot as plt

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_FILENAME = 'cifar-10-python.tar.gz'
DATA_PATH = '../input/'

# --------------- Loading Data-set ---------------

def pickle_load(f):
      with open(f, 'rb') as pickle_file:
            return cPickle.load(pickle_file)

def _dl_progress(count, blockSize, totalSize):
      """ Simple download progress indicator """
      percent = int(count * blockSize * 100 / totalSize)
      sys.stdout.write("\r" + DATA_FILENAME + " ...[%d%%]" % percent)
      sys.stdout.flush()
      
def _load_data(url=DATA_URL, filename=DATA_FILENAME, path=DATA_PATH):
    """Load data from `url` and store the result in `filename`."""
    destfile = os.path.join(path, filename)
    if not os.path.exists(destfile):
        print("Downloading CIFAR-10 dataset")
        urlretrieve(url, destfile, reporthook=_dl_progress)

    archive = tarfile.open(destfile)
    archive.extractall(path)
    archive.close()

def load_data(path=DATA_PATH):
    """
    Get and normalize data with training and test sets.
    Automatically shuffles the training batch.
    
    A single data_batch file is organized as follows:

    10000 x 1024 NumPy array of grayscale image values
    10000        NumPy array of numerical labels
    10000        list of image file names

    """
    data = _load_data()
    X, y = [], []
    
    src = os.path.join(path, 'cifar-10-python/')
    for i in range(5):
        data_train = pickle_load(src+'data_batch_'+`i+1`)
        X.append(data_train['data'])
        y.append(data_train['labels'])

    data_test = pickle_load(src+'test_batch')
        
    X_train = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
    X_test = data_test['data'].reshape(-1, 3, 32, 32).astype(np.float32)

    y_train = np.concatenate(y).astype(np.int32)
    y_test = np.array(data_test['labels']).astype(np.int32)

    X_train -= X_train.mean()
    X_train /= X_train.std()

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    
    return X_train, y_train, X_test, y_test

# --------------- Network architectures ---------------

deep_convnet = [
    (layers.InputLayer, {'shape': (None, 3, 32, 32)}),

    (layers.Conv2DLayer, {'num_filters': 8, 'filter_size': (5, 5), 'pad': 2}),
    (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),

    (layers.Conv2DLayer, {'num_filters': 16, 'filter_size': (5, 5), 'pad': 2}),
    (layers.Conv2DLayer, {'num_filters': 16, 'filter_size': (5, 5), 'pad': 2}),
    (layers.Conv2DLayer, {'num_filters': 16, 'filter_size': (5, 5), 'pad': 2}),
    (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),

    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (2, 2)}),
    (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),

    (layers.DenseLayer, {'num_units': 1000, 'nonlinearity': rectify}),
    (layers.DropoutLayer, {'p': 0.2}),
    (layers.DenseLayer, {'num_units': 100, 'nonlinearity': softmax}),

    ]

network_in_network = [
        (layers.InputLayer,  {'shape': (None, 3, 32, 32)}),

        (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': 5, 'pad': 2, 'stride': 1}),
        (layers.NINLayer, {'num_units': 32}),
        (layers.NINLayer, {'num_units': 16}),
        (layers.DropoutLayer, {'p': 0.5}),

        (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': 5, 'pad': 2, 'stride': 1}),
        (layers.NINLayer, {'num_units': 64}),
        (layers.NINLayer, {'num_units': 64}),
        (layers.DropoutLayer, {'p': 0.5}),
        
        (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': 5, 'pad': 2, 'stride': 1}),
        (layers.NINLayer, {'num_units': 64}),
        (layers.NINLayer, {'num_units': 10}),

        (layers.GlobalPoolLayer, {}),
        (layers.DenseLayer, {'num_units':  10, 'nonlinearity': softmax})       
        ]

# ---------------  Building the network ---------------

def float32(k):
    """
    Converts numbers into 32b floats best used w/ GPUs
    """
    return np.cast['float32'](k)

class AdjustVariable(object):
    """
    On-the-fly tweaking of network's hyperparameters
    (learning rate and momentum) during the training.

    Credits to (@karpathy)
    """
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

class EarlyStopping(object):
    """
    Will stop the training and load back the best weight
    configuration when no longer improving after a given
    period of time (patience). Prevents overfitting.

    Credits to (@dnouri)
    """
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

        
def build_network(layers):
    """
    Builds and returns a NeuralNet object
    """
    return  NeuralNet(
        layers = layers,

        update = nesterov_momentum,
        update_learning_rate = theano.shared(float32(0.01)),
        update_momentum = theano.shared(float32(0.9)),    

        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=200),
            ],

        # Automatically split the data (90% training 10% validation).
        train_split = TrainSplit(eval_size=0.10), 

        regression = False,
        max_epochs = 300,
        verbose = 2, 
        )

# --------------- Training the network ---------------

def display_data(path='../input/cifar-10-python/data_batch_1'):
    """
    Plot 3 random grayscale CIFAR-10 images with and their respective name and label.
    """

    batch = pickle_load(path)
    imgdata = batch['data']    
    # Combine the R, G, and B components together to get grayscale, using the luminosity-preserving formula:
    # see http://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale
    # source: http://home.wlu.edu/~levys/courses/csci315w2016/assignments/cifar10.py (@ Simon D. Levy)
    grayscale = 0.21*imgdata[:,0:1024] + 0.72*imgdata[:,1024:2048] + 0.07*imgdata[:,2048:3072]

    images = grayscale
    labels = np.array(batch['labels'])
    names = batch['filenames']

    plt.figure(figsize=(16,5))
    for i in range(3):
        index = random.randint(0, len(images))
        example = images[index].reshape(32, 32)
        name = names[index]
        label = labels[index    ]
        plt.subplot(1, 3, i+1)
        plt.imshow(example, cmap='gray', interpolation='nearest')
        plt.title("Name: {0}\nLabel: {1}".format(name, label))
        plt.axis("off")
    plt.show()

    
def main():
    
    X_train, y_train, X_test, y_test = load_data()
    net = build_network(deep_convnet)
    display_data()
    net.fit(X_train, y_train)

    
if __name__ == '__main__':
    main()
