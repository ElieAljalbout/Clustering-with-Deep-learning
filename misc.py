'''
Created on Jul 11, 2017
'''

import cPickle
import gzip

import numpy as np
from PIL import Image
import matplotlib

# For plotting graphs via ssh with no display
# Ref: https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from numpy import float32
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans
from sklearn import manifold
from sklearn.utils.linear_assignment_ import linear_assignment


class DatasetHelper(object):
    '''
    Utility class for handling different datasets
    '''

    def __init__(self, name):
        '''
        A dataset instance keeps dataset name, the input set, the flat version of input set
        and the cluster labels
        '''
        self.name = name
        if name == 'MNIST':
            self.dataset = MNISTDataset()
        elif name == 'STL':
            self.dataset = STLDataset()
        elif name == 'COIL20':
            self.dataset = COIL20Dataset()

    def loadDataset(self):
        '''
        Load the appropriate dataset based on the dataset name
        '''
        self.input, self.labels, self.input_flat = self.dataset.loadDataset()

    def getClusterCount(self):
        '''
        Number of clusters in the dataset - e.g 10 for mnist, 20 for coil20
        '''
        return self.dataset.cluster_count

    def iterate_minibatches(self, set_type, batch_size, targets=None, shuffle=False):
        '''
        Utility method for getting batches out of a dataset
        :param set_type: IMAGE - suitable input for CNNs or FLAT - suitable for DNN
        :param batch_size: Size of minibatches
        :param targets: None if the output should be same as inputs (autoencoders), otherwise takes a target array from which batches can be extracted. Must have the same order as the dataset, e.g, dataset inputs nth sample has output at target's nth element
        :param shuffle: If the dataset needs to be shuffled or not
        :return: generates a batches of size batch_size from the dataset, each batch is the pair (input, output)
        '''
        inputs = None
        if set_type == 'IMAGE':
            inputs = self.input
            if targets is None:
                targets = self.input
        elif set_type == 'FLAT':
            inputs = self.input_flat
            if targets is None:
                targets = self.input_flat
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]


class MNISTDataset(object):
    '''
    Class for reading and preparing MNIST dataset
    '''

    def __init__(self):
        self.cluster_count = 10

    def loadDataset(self):
        f = gzip.open('mnist/mnist.pkl.gz', 'rb')
        train_set, _, test_set = cPickle.load(f)
        train_input, train_input_flat, train_labels = self.prepareDatasetForAutoencoder(train_set[0], train_set[1])
        test_input, test_input_flat, test_labels = self.prepareDatasetForAutoencoder(test_set[0], test_set[1])
        f.close()
        # combine test and train samples
        return [np.concatenate((train_input, test_input)), np.concatenate((train_labels, test_labels)),
                np.concatenate((train_input_flat, test_input_flat))]

    def prepareDatasetForAutoencoder(self, inputs, targets):
        '''
        Returns the image, flat and labels as a tuple
        '''
        X = inputs
        X = X.reshape((-1, 1, 28, 28))
        return (X, X.reshape((-1, 28 * 28)), targets)


class STLDataset(object):
    '''
    Class for preparing and reading the STL dataset
    '''

    def __init__(self):
        self.cluster_count = 10

    def loadDataset(self):
        train_x = np.fromfile('stl/train_X.bin', dtype=np.uint8)
        train_y = np.fromfile('stl/train_y.bin', dtype=np.uint8)
        test_x = np.fromfile('stl/train_X.bin', dtype=np.uint8)
        test_y = np.fromfile('stl/train_y.bin', dtype=np.uint8)
        train_input = np.reshape(train_x, (-1, 3, 96, 96))
        train_labels = train_y
        train_input_flat = np.reshape(test_x, (-1, 1, 3 * 96 * 96))
        test_input = np.reshape(test_x, (-1, 3, 96, 96))
        test_labels = test_y
        test_input_flat = np.reshape(test_x, (-1, 1, 3 * 96 * 96))
        return [np.concatenate(train_input, test_input), np.concatenate(train_labels, test_labels),
                np.concatenate(train_input_flat, test_input_flat)]


class COIL20Dataset(object):
    '''
    Class for reading and preparing the COIL20Dataset
    '''

    def __init__(self):
        self.cluster_count = 20

    def loadDataset(self):
        train_x = np.load('coil/coil_X.npy').astype(np.float32) / 256.0
        train_y = np.load('coil/coil_y.npy')
        train_x_flat = np.reshape(train_x, (-1, 128 * 128))
        return [train_x, train_y, train_x_flat]


def rescaleReshapeAndSaveImage(image_sample, out_filename):
    '''
    For saving the reconstructed output as an image
    :param image_sample: output of the autoencoder
    :param out_filename: filename for the saved image
    :return: None (side effect) Image saved
    '''
    image_sample = ((image_sample - np.amin(image_sample)) / (np.amax(image_sample) - np.amin(image_sample))) * 255;
    image_sample = np.rint(image_sample).astype(int)
    image_sample = np.clip(image_sample, a_min=0, a_max=255).astype('uint8')
    img = Image.fromarray(image_sample, 'L')
    img.save(out_filename)


def cluster_acc(y_true, y_pred):
    '''
    Uses the hungarian algorithm to find the best permutation mapping and then calculates the accuracy wrt
    Implementation inpired from https://github.com/piiswrong/dec, since scikit does not implement this metric
    this mapping and true labels
    :param y_true: True cluster labels
    :param y_pred: Predicted cluster labels
    :return: accuracy score for the clustering
    '''
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(y_pred.size):
        idx1 = int(y_pred[i])
        idx2 = int(y_true[i])
        w[idx1, idx2] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def getClusterMetricString(method_name, labels_true, labels_pred):
    '''
    Creates a formatted string containing the method name and acc, nmi metrics - can be used for printing
    :param method_name: Name of the clustering method (just for printing)
    :param labels_true: True label for each sample
    :param labels_pred: Predicted label for each sample
    :return: Formatted string containing metrics and method name
    '''
    acc = cluster_acc(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    return '%-50s     %8.3f     %8.3f' % (method_name, acc, nmi)


def evaluateKMeans(data, labels, nclusters, method_name):
    '''
    Clusters data with kmeans algorithm and then returns the string containing method name and metrics, and also the evaluated cluster centers
    :param data: Points that need to be clustered as a numpy array
    :param labels: True labels for the given points
    :param nclusters: Total number of clusters
    :param method_name: Name of the method from which the clustering space originates (only used for printing)
    :return: Formatted string containing metrics and method name, cluster centers
    '''
    kmeans = KMeans(n_clusters=nclusters, n_init=20)
    kmeans.fit(data)
    return getClusterMetricString(method_name, labels, kmeans.labels_), kmeans.cluster_centers_


def visualizeData(Z, labels, num_clusters, title):
    '''
    TSNE visualization of the points in latent space Z
    :param Z: Numpy array containing points in latent space in which clustering was performed
    :param labels: True labels - used for coloring points
    :param num_clusters: Total number of clusters
    :param title: filename where the plot should be saved
    :return: None - (side effect) saves clustering visualization plot in specified location
    '''
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=fig.dpi)
