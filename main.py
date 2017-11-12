'''
Created on Jul 9, 2017
'''
import numpy
import json
from misc import DatasetHelper, evaluateKMeans, visualizeData
from network import DCJC, rootLogger
import argparse


def testOnlyClusterInitialization(dataset_name, arch, epochs):
    '''
    Train an autoencoder defined by architecture arch and trains it with the dataset defined
    :param dataset_name: Name of the dataset with which the network will be trained [MNIST, COIL20]
    :param arch: Architecture of the network as a dictionary. Specification for architecture can be found in readme.md
    :param epochs: Number of train epochs
    :return: None - (side effect) saves the latent space and params of trained network in an appropriate location in saved_params folder
    '''
    rootLogger.info("Loading dataset")
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info("Done loading dataset")
    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    rootLogger.info("Done creating network")
    rootLogger.info("Starting training")
    dcjc.pretrainWithData(dataset, epochs, False);


def testOnlyClusterImprovement(dataset_name, arch, epochs, method):
    '''
    Use an initialized autoencoder and train it along with clustering loss. Assumed that pretrained autoencoder params
    are available, i.e. testOnlyClusterInitialization has been run already with the given params
    :param dataset_name: Name of the dataset with which the network will be trained [MNIST, COIL20]
    :param arch: Architecture of the network as a dictionary. Specification for architecture can be found in readme.md
    :param epochs: Number of train epochs
    :param method: Can be KM or KLD - depending on whether the clustering loss is KLDivergence loss between the current KMeans distribution(Q) and a more desired one(Q^2), or if the clustering loss is just the Kmeans loss
    :return: None - (side effect) saves latent space and params of the trained network
    '''
    rootLogger.info("Loading dataset")
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info("Done loading dataset")
    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    rootLogger.info("Starting cluster improvement")
    if method == 'KM':
        dcjc.doClusteringWithKMeansLoss(dataset, epochs)
    elif method == 'KLD':
        dcjc.doClusteringWithKLdivLoss(dataset, True, epochs)


def testKMeans(dataset_name, archs):
    '''
    Performs kMeans clustering, and report metrics on the output latent space produced by the networks defined in archs,
    with given dataset. Assumes that testOnlyClusterInitialization and testOnlyClusterImprovement have been run before
    this for the specified archs/datasets, as the results saved by them are used for clustering
    :param dataset_name: Name of dataset [MNIST, COIL20]
    :param archs: Architectures as a dictionary
    :return: None - reports the accuracy and nmi clustering metrics
    '''
    rootLogger.info('Initial Cluster Quality Comparison')
    rootLogger.info(80 * '_')
    rootLogger.info('%-50s     %8s     %8s' % ('method', 'ACC', 'NMI'))
    rootLogger.info(80 * '_')
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info(evaluateKMeans(dataset.input_flat, dataset.labels, dataset.getClusterCount(), 'image')[0])
    for arch in archs:
        Z = numpy.load('saved_params/' + dataset.name + '/z_' + arch['name'] + '.npy')
        rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), arch['name'])[0])
        Z = numpy.load('saved_params/' + dataset.name + '/pc_z_' + arch['name'] + '.npy')
        rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), arch['name'])[0])
        Z = numpy.load('saved_params/' + dataset.name + '/pc_km_z_' + arch['name'] + '.npy')
        rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), arch['name'])[0])
    rootLogger.info(80 * '_')


def visualizeLatentSpace(dataset_name, arch):
    '''
    Plots and saves graphs for visualized images space, autoencoder latent space, and the final clustering latent space
    :param dataset_name: Name of dataset [MNIST, COIL20]
    :param arch: Architectures as a dictionary
    :return: None - (side effect) saved graphs in plots/ folder
    '''
    rootLogger.info("Loading dataset")
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info("Done loading dataset")
    # We consider only the first 5000 point or less for better visualization
    max_points = min(dataset.input_flat.shape[0], 5000)
    # Image space
    visualizeData(dataset.input_flat[0:max_points], dataset.labels[0:max_points], dataset.getClusterCount(), "plots/%s/raw.png" % dataset.name)
    # Latent space - autoencoder
    Z = numpy.load('saved_params/' + dataset.name + '/z_' + arch['name'] + '.npy')
    visualizeData(Z[0:max_points], dataset.labels[0:max_points], dataset.getClusterCount(), "plots/%s/autoencoder.png" % dataset.name)
    # Latent space - kl div clustering network
    Z = numpy.load('saved_params/' + dataset.name + '/pc_z_' + arch['name'] + '.npy')
    visualizeData(Z[0:max_points], dataset.labels[0:max_points], dataset.getClusterCount(), "plots/%s/clustered_kld.png" % dataset.name)
    # Latent space - kmeans clustering network
    Z = numpy.load('saved_params/' + dataset.name + '/pc_km_z_' + arch['name'] + '.npy')
    visualizeData(Z[0:max_points], dataset.labels[0:max_points], dataset.getClusterCount(), "plots/%s/clustered_km.png" % dataset.name)


if __name__ == '__main__':
    '''
    usage: main.py [-h] -d DATASET -a ARCHITECTURE [--pretrain PRETRAIN]
               [--cluster CLUSTER] [--metrics METRICS] [--visualize VISUALIZE]
        
    required arguments:
      -d DATASET, --dataset DATASET
                            Dataset on which autoencoder is trained [MNIST,COIL20]
      -a ARCHITECTURE, --architecture ARCHITECTURE
                            Index of architecture of autoencoder in the json file
                            (archs/)
                            
    optional arguments:
      -h, --help            show this help message and exit
      --pretrain PRETRAIN   Pretrain the autoencoder for specified #epochs
                            specified by architecture on specified dataset
      --cluster CLUSTER     Refine the autoencoder for specified #epochs with
                            clustering loss, assumes that pretraining results are
                            available
      --metrics METRICS     Report k-means clustering metrics on the clustered
                            latent space, assumes pretrain and cluster based
                            training have been performed
      --visualize VISUALIZE
                            Visualize the image space and latent space, assumes
                            pretraining and cluster based training have been
                            performed
    '''
    # Load architectures from the json files
    mnist_archs = []
    coil_archs = []
    with open("archs/coil.json") as archs_file:
        coil_archs = json.load(archs_file)
    with open("archs/mnist.json") as archs_file:
        mnist_archs = json.load(archs_file)
    # Argument parsing
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('required arguments')
    requiredArgs.add_argument("-d", "--dataset", help="Dataset on which autoencoder is trained [MNIST,COIL20]", required=True)
    requiredArgs.add_argument("-a", "--architecture", type=int, help="Index of architecture of autoencoder in the json file (archs/)", required=True)
    parser.add_argument("--pretrain", type=int, help="Pretrain the autoencoder for specified #epochs specified by architecture on specified dataset")
    parser.add_argument("--cluster", type=int, help="Refine the autoencoder for specified #epochs with clustering loss, assumes that pretraining results are available")
    parser.add_argument("--metrics", action='store_true', help="Report k-means clustering metrics on the clustered latent space, assumes pretrain and cluster based training have been performed")
    parser.add_argument("--visualize", action='store_true', help="Visualize the image space and latent space, assumes pretraining and cluster based training have been performed")
    args = parser.parse_args()
    # Train/Visualize as per the arguments
    dataset_name = args.dataset
    arch_index = args.architecture
    if dataset_name == 'MNIST':
        archs = mnist_archs
    elif dataset_name == 'COIL20':
        archs = coil_archs
    if args.pretrain:
        testOnlyClusterInitialization(dataset_name, archs[arch_index], args.pretrain)
    if args.cluster:
        testOnlyClusterImprovement(dataset_name, archs[arch_index], args.cluster, "KLD")
    if args.metrics:
        testKMeans(dataset_name, [archs[arch_index]])
    if args.visualize:
        visualizeLatentSpace(dataset_name, archs[arch_index])
