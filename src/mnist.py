from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, rectify
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from nolearn.lasagne import visualize

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from urllib import urlretrieve

import os
import sys
import gzip
import cPickle

import theano
import lasagne
import numpy as np
import matplotlib.pyplot as plt
    
DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'
<<<<<<< HEAD
DATA_PATH = '../input/mnist/'
=======
DATA_PATH = '../input/'
>>>>>>> 370a4ae3d758958d90fbf421041efc31afed7206

# -------------------- Loading Data-set --------------------

def pickle_load(f, encoding):
    return cPickle.load(f)

def _dl_progress(count, blockSize, totalSize):
      """ Simple download progress indicator """
      percent = int(count*blockSize*100/totalSize)
      sys.stdout.write("\r" + DATA_FILENAME + "...%d%%" % percent)
      sys.stdout.flush()

def _load_data(url=DATA_URL, filename=DATA_FILENAME, path=DATA_PATH):
    """Load data from `url` and store the result in `filename`."""
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        destfile = os.path.join(path, filename)
        urlretrieve(url, destfile, reporthook=_dl_progress)

    with gzip.open(destfile, 'rb') as f:
        return cPickle.load(f)

def load_data():
    """
    Get and normalize data with labels, split into training, validation and test set.
    Automatically shuffles the training batch.
    """
    data = _load_data()
    
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    X_train = X_train.reshape((-1, 1, 28, 28)).astype(np.float32)
    X_valid = X_valid.reshape((-1, 1, 28, 28)).astype(np.float32)
    X_test = X_test.reshape((-1, 1, 28, 28)).astype(np.float32)
    
    y_train = y_train.astype(np.int32)
    y_valid = y_valid.astype(np.int32)
    y_test = y_test.astype(np.int32)

    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# --------------- Network architectures ---------------

"""
Write different network implementations to use when
calling build_network()
"""
    
dropout_net = [
    (layers.InputLayer, {'shape': (None, 1, 26, 26)}),

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
    (layers.InputLayer,  {'shape': (None, 1, 26, 26)}),

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

# -------------------- Building the network -------------------

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
        
            
class CropBatchIterator(BatchIterator):
    """
    Custom Batch Iterator called by NeuralNet to provided augmented
    data in order to reduce overfitting. Will randomly crop the inputed image
    by 2 pixels (on-the-fly and CPU driven).
    """
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
    return NeuralNet(
        layers = layers,

        batch_iterator_train = CropBatchIterator(batch_size=256),

        update = nesterov_momentum,
        update_learning_rate = theano.shared(float32(0.03)),
        update_momentum = theano.shared(float32(0.9)),    

        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=10),
            ],

        regression = False,
        max_epochs = 100,
        verbose = 2
        )

# --------------- Training the network ---------------

def display_data(X, y):
    """
    Plot a 4*4 matrix with MNIST digits and their respective labels.
    Helps detecting if the shuffle correctly occured.
    """
    figs, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(-X[i + 4 * j].reshape(28, 28), cmap='gray', interpolation='none')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
            axes[i, j].axis('off')
    plt.show()

def display_confusion_matrix(test_data, test_labels, save=False):
    """
    Plot a matrix representing the choices made by the network
    on a testing batch.
    X axis are the predicted values,
    Y axis are the expected values.

    If the flag save is set to True, the output will be saved
    in a .png image.
    """
    expected = test_labels
    predicted = mnist.predict(test_data)
    cm = confusion_matrix(expected, predicted)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('Expected label')
    plt.xlabel('Predicted label')
    plt.show()
    if save is True:
        plt.savefig("../results/mnist/confusion_matrix.png")

        
def main():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    net = build_network(dropout_net)
    net.fit(X_train, y_train)

    display_data(X_train, y_train)
    display_confusion_matrix(X_test, y_test)

    
if __name__ == '__main__':
    main()
