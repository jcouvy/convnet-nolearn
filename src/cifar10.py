import cPickle
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '/net/www/jcouvy/data/'

# --------------- Loading Data-set ---------------

def download_data():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    if not os.path.exists(filename):
        print("Downloading CIFAR-10 dataset...")
        urlretrieve(url, filename)

def unpickle(file):
    with gzip.open(file, 'rb') as f:
        dict = cPickle.load(f)
        f.close()
    return dict

def load_data():
    X, y = [], []
    
    for i in range(5):
        batch = unpickle(DATA_PATH+'data_batch_'+`i+1`)
        X.append(batch['data'])
        y.append(batch['labels'])
        
    X_train = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
    y_train = np.concatenate(y).astype(np.int32)

    X_train -= X_train.mean()
    X_train /= X_train.std()

    return X_train, y_train

# -------------------- Building the network -------------------

import theano

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, rectify

from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo, TrainSplit
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights

def float32(k):
    return np.cast['float32'](k)


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

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

        return Xb, yb

    
cifar10 = NeuralNet(
    
    layers = [
        (layers.InputLayer,  {'shape': (None, 3, 32, 32)}),

        (layers.Conv2DLayer, {'num_filters': 20, 'filter_size': 3}),
        (layers.MaxPool2DLayer, {'pool_size': 3, 'stride': 2}),
        
        (layers.Conv2DLayer, {'num_filters': 20, 'filter_size': 3}),
        (layers.MaxPool2DLayer, {'pool_size': 2, 'stride': 2}),

        (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 2}),
        (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 2}),
        (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 2}),
        (layers.MaxPool2DLayer, {'pool_size': 2, 'stride': 2}),
        
        (layers.DenseLayer, {'num_units': 1024, 'nonlinearity': rectify}),
        (layers.DropoutLayer, {'p': 0.5}),
        (layers.DenseLayer, {'num_units':  100, 'nonlinearity': softmax})        
    ],

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(0.03)),
    update_momentum = theano.shared(float32(0.9)),    

    # NeuralNet allows us to dynamically tune its parameters. We use the class
    # AdjustVariable to decrease the learning rate with the number of epochs
    # while the momentum increases.
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=250),
        ],

    batch_iterator_train = FlipBatchIterator(batch_size=128),
        
    regression = False,
    train_split = TrainSplit(eval_size=0.20),
    max_epochs = 500,
    verbose = 2, 
)

X_train, y_train = load_data()
np.random.seed(1234)

cifar10.fit(X_train, y_train)

plot_loss(cifar10)
plt.savefig("../results/plotloss.png")

plot_conv_weights(cifar10.layers_[1], figsize=(4, 4))
plt.savefig("../results/convweights_l1.png")

plot_conv_weights(cifar10.layers_[7], figsize=(4, 4))
plt.savefig("../results/convweights_l7.png")
