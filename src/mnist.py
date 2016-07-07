# -------------------- Loading Data-set --------------------

import cPickle
import pandas as pd
import numpy as np
import sys

# The competition datafiles are in the directory ../input
# Read training and test data files.
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

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

# -------------------- Building the network -------------------

import theano
import lasagne
from lasagne import layers
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, rectify
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo, TrainSplit

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
        
def dict_slice(arr, sl):
    """
    Helper method to slice all arrays contained in a dictionary.
    """
    if isinstance(arr, dict):
        ret = OrderedDict()
        for k, v in arr.items():
            ret[k] = v[sl]
        return ret
    else:
        return arr[sl]

def list_slice(arr, sl):
    """
    Helper method to slice all arrays contained in a list.
    """
    if isinstance(arr, list) or isinstance(arr, tuple):
        ret = []
        for v in arr:
            ret.append(v[sl])
        return ret
    else:
        return arr[sl]

class AlignedBatchIterator(object):
    """
        A simple iterator class, accepts an arbitrary number of numpy arrays.
        
        Assumes that all numpy arrays are of equal length.
        
        Inspired by the BatchIterator class used in nolearn.
        https://github.com/dnouri/nolearn
        Copyright (c) 2012-2015 Daniel Nouri
        
        Parameters
        ----------
        batch_size : int
            Size of mini batch
        shuffle : bool, optional
            Shuffle data before iterating
    """
    
    def __init__(self, batch_size, shuffle=True):
        """
        batch_size - size of every minibatch 
        shuffle    - whether to shuffle data (default is true)
        """
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.elem_list = None

    def __call__(self, *args):
        """
        Note for developers:
        The __call__ magic function puts all passed arguments into a list
        elem_list which is used for iteration.
        
        It also checks whether all args contain the same number of elements.
        """
        self.elem_list = args

        
        return self

    def __iter__(self):
        bs = self.batch_size
        indices = range(len(self.elem_list[0]))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range((self.n_samples + bs - 1) // bs):
            sl = indices[slice(i * bs, (i + 1) * bs)]
            belem_dict = list_slice(self.elem_list, sl)
            yield belem_dict

    @property
    def n_samples(self):
        X = self.elem_list[0]
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    @property
    def num_inputs(self):
        return len(self.elem_list)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('elem_list',):
            if attr in state:
                del state[attr]
        return state

            
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

        
dropout_net = [
    (layers.InputLayer, {'shape':(None, 1, 28, 28)}),

    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 3}),
    (layers.MaxPool2DLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.3}),

    (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': 3}),
    (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': 3}),
    (layers.MaxPool2DLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.3}),

    (layers.DenseLayer, {'num_units': 1000, 'nonlinearity': rectify}),
    (layers.DropoutLayer, {'p': 0.3}),
    (layers.DenseLayer, {'num_units': 100,   'nonlinearity': softmax}),
    ]

    
deep_convnet = [
    (layers.InputLayer, {'shape': (None, 1, 28, 28)}),

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

def build_network(layers):    
    return NeuralNet(
        layers = layers,

        batch_iterator_train = AlignedBatchIterator(batch_size=128, shuffle=True),

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

mnist = build_network(deep_convnet)
mnist.fit(X_train, y_train)
mnist.predict(X_test)

