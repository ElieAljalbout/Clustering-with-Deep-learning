'''
Created on Jul 11, 2017
'''

from datetime import datetime
import logging

from lasagne import layers
import lasagne
from lasagne.layers.helper import get_all_layers
import theano
import signal

from customlayers import ClusteringLayer, Unpool2DLayer, getSoftAssignments
from misc import evaluateKMeans, visualizeData, rescaleReshapeAndSaveImage
import numpy as np
import theano.tensor as T

from lasagne.layers import batch_norm

# Logging utilities - logs get saved in folder logs named by date and time, and also output
# at standard output

logFormatter = logging.Formatter("[%(asctime)s]  %(message)s", datefmt='%m/%d %I:%M:%S')

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(datetime.now().strftime('logs/dcjc_%H_%M_%d_%m.log'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


class DCJC(object):
    # Main class holding autoencoder network and training functions
    def __init__(self, network_description):

        signal.signal(signal.SIGINT, self.signal_handler)
        self.name = network_description['name']
        netbuilder = NetworkBuilder(network_description)
        self.shouldStopNow  = False
        # Get the lasagne network using the network builder class that creates autoencoder with the specified architecture
        self.network = netbuilder.buildNetwork()
        self.encode_layer, self.encode_size = netbuilder.getEncodeLayerAndSize()
        self.t_input, self.t_target = netbuilder.getInputAndTargetVars()
        self.input_type = netbuilder.getInputType()
        self.batch_size = netbuilder.getBatchSize()
        rootLogger.info("Network: " + self.networkToStr())
        # Reconstruction is just output of the network
        recon_prediction_expression = layers.get_output(self.network)
        # Latent/Encoded space is the output of the bottleneck/encode layer
        encode_prediction_expression = layers.get_output(self.encode_layer, deterministic=True)
        # Loss for autoencoder = reconstruction loss + weight decay regularizer
        loss = self.getReconstructionLossExpression(recon_prediction_expression, self.t_target)
        weightsl2 = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
        loss += (5e-5 * weightsl2)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        # SGD with momentum + Decaying learning rate
        self.learning_rate = theano.shared(lasagne.utils.floatX(0.01))
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=self.learning_rate)
        # Theano functions for calculating loss, predicting reconstruction, encoding
        self.trainAutoencoder = theano.function([self.t_input, self.t_target], loss, updates=updates)
        self.predictReconstruction = theano.function([self.t_input], recon_prediction_expression)
        self.predictEncoding = theano.function([self.t_input], encode_prediction_expression)

    def getReconstructionLossExpression(self, prediction_expression, t_target):
        '''
        Reconstruction loss = means square error between input and reconstructed input
        '''
        loss = lasagne.objectives.squared_error(prediction_expression, t_target)
        loss = loss.mean()
        return loss

    def signal_handler(self,signal, frame):

        command = raw_input('\nWhat is your command?')
        if str(command).lower()=="stop":
            self.shouldStopNow  = True
        else:
            exec(command)

    def pretrainWithData(self, dataset, epochs, continue_training=False):
        '''
        Pretrains the autoencoder on the given dataset
        :param dataset: Data on which the autoencoder is trained
        :param epochs: number of training epochs
        :param continue_training: Resume training if saved params available
        :return: None - (side effect) saves the trained network params and latent space in appropriate location
        '''
        batch_size = self.batch_size
        # array for holding the latent space representation of input
        Z = np.zeros((dataset.input.shape[0], self.encode_size), dtype=np.float32);
        # in case we're continuing training load the network params
        if continue_training:
            with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name)) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(self.network, param_values, trainable=True)
        for epoch in range(epochs):
            error = 0
            total_batches = 0
            for batch in dataset.iterate_minibatches(self.input_type, batch_size, shuffle=True):
                inputs, targets = batch
                error += self.trainAutoencoder(inputs, targets)
                total_batches += 1
            # learning rate decay
            self.learning_rate.set_value(self.learning_rate.get_value() * lasagne.utils.floatX(0.9999))
            # For every 20th iteration, print the clustering accuracy and nmi - for checking if the network
            # is actually doing something meaningful - the labels are never used for training
            if (epoch + 1) % 2 == 0:
                for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                    Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
                    # Uncomment the next two lines to create reconstruction outputs in folder dumps/ (may need to be created)
                    #for i, x in enumerate(self.predictReconstruction(batch[0])):
                    #	print('dump')
                    #	rescaleReshapeAndSaveImage(x[0], "dumps/%02d%03d.jpg"%(epoch,i));
                rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches))[0])
            else:
                # Just report the training loss
                rootLogger.info("%-30s     %8s     %8s" % ("%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches), "", ""))
            if self.shouldStopNow:
            	break
        # The inputs in latent space after pretraining
        for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
            Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
        # Save network params and latent space
        np.save('saved_params/%s/z_%s.npy' % (dataset.name, self.name), Z)
        # Borrowed from mnist lasagne example
        np.savez('saved_params/%s/m_%s.npz' % (dataset.name, self.name), *lasagne.layers.get_all_param_values(self.network, trainable=True))

    def doClusteringWithKLdivLoss(self, dataset, combined_loss, epochs):
        '''
        Trains the autoencoder with combined kldivergence loss and reconstruction loss, or just the kldivergence loss
        At the moment does not give good results
        :param dataset: Data on which the autoencoder is trained
        :param combined_loss: boolean - whether to use both reconstruction and kl divergence loss or just kldivergence loss
        :param epochs: Number of training epochs
        :return: None - (side effect) saves the trained network params and latent space in appropriate location
        '''
        batch_size = self.batch_size
        # Load saved network params and inputs in latent space obtained after pretraining
        with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values, trainable=True)
        Z = np.load('saved_params/%s/z_%s.npy' % (dataset.name, self.name))
        # Find initial cluster centers
        quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), 'Initial')
        rootLogger.info(quality_desc)
        # P is the more pure target distribution we want to achieve
        P = T.matrix('P')
        # Extend the network so it calculates soft assignment cluster distribution for the inputs in latent space
        clustering_network = ClusteringLayer(self.encode_layer, dataset.getClusterCount(), cluster_centers, batch_size,self.encode_size)
        soft_assignments = layers.get_output(clustering_network)
        reconstructed_output_exp = layers.get_output(self.network)
        # Clustering loss = kl divergence between the pure distribution P and current distribution
        clustering_loss = self.getKLDivLossExpression(soft_assignments, P)
        reconstruction_loss = self.getReconstructionLossExpression(reconstructed_output_exp, self.t_target)
        params_ae = lasagne.layers.get_all_params(self.network, trainable=True)
        params_dec = lasagne.layers.get_all_params(clustering_network, trainable=True)
        # Total loss = weighted sum of the two losses
        w_cluster_loss = 1
        w_reconstruction_loss = 1
        total_loss = w_cluster_loss * clustering_loss
        if (combined_loss):
            total_loss = total_loss + w_reconstruction_loss * reconstruction_loss
        all_params = params_dec
        if combined_loss:
            all_params.extend(params_ae)
        # Parameters = unique parameters in the new network
        all_params = list(set(all_params))
        # SGD with momentum, LR = 0.01, Momentum = 0.9
        updates = lasagne.updates.nesterov_momentum(total_loss, all_params, learning_rate=0.01)
        # Function to calculate the soft assignment distribution
        getSoftAssignments = theano.function([self.t_input], soft_assignments)
        # Train function - based on whether complete loss is used or not
        trainFunction = None
        if combined_loss:
            trainFunction = theano.function([self.t_input, self.t_target, P], total_loss, updates=updates)
        else:
            trainFunction = theano.function([self.t_input, P], clustering_loss, updates=updates)
        for epoch in range(epochs):
            # Get the current distribution
            qij = np.zeros((dataset.input.shape[0], dataset.getClusterCount()), dtype=np.float32)
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                qij[i * batch_size: (i + 1) * batch_size] = getSoftAssignments(batch[0])
            # Calculate the desired distribution
            pij = self.calculateP(qij)
            error = 0
            total_batches = 0
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, pij, shuffle=True)):
                if (combined_loss):
                    error += trainFunction(batch[0], batch[0], batch[1])
                else:
                    error += trainFunction(batch[0], batch[1])
                total_batches += 1
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
            # For every 10th iteration, print the clustering accuracy and nmi - for checking if the network
            # is actually doing something meaningful - the labels are never used for training
            if (epoch + 1) % 10 == 0:
                rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d [%.4f]" % (
                    epoch, error / total_batches))[0])
            if self.shouldStopNow:
           	   break
        # Save the inputs in latent space and the network parameters
        for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
            Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
        np.save('saved_params/%s/pc_z_%s.npy' % (dataset.name, self.name), Z)
        np.savez('saved_params/%s/pc_m_%s.npz' % (dataset.name, self.name),
                 *lasagne.layers.get_all_param_values(self.network, trainable=True))

    def calculateP(self, Q):
        # Function to calculate the desired distribution Q^2, for more details refer to DEC paper
        f = Q.sum(axis=0)
        pij_numerator = Q * Q
        pij_numerator = pij_numerator / f
        normalizer_p = pij_numerator.sum(axis=1).reshape((Q.shape[0], 1))
        P = pij_numerator / normalizer_p
        return P

    def getKLDivLossExpression(self, Q_expression, P_expression):
        # Loss = KL Divergence between the two distributions
        log_arg = P_expression / Q_expression
        log_exp = T.log(log_arg)
        sum_arg = P_expression * log_exp
        loss = sum_arg.sum(axis=1).sum(axis=0)
        return loss

    def doClusteringWithKMeansLoss(self, dataset, epochs):
        '''
        Trains the autoencoder with combined kMeans loss and reconstruction loss
        At the moment does not give good results
        :param dataset: Data on which the autoencoder is trained
        :param epochs: Number of training epochs
        :return: None - (side effect) saves the trained network params and latent space in appropriate location
        '''
        batch_size = self.batch_size
        # Load the inputs in latent space produced by the pretrained autoencoder and use it to initialize cluster centers
        Z = np.load('saved_params/%s/z_%s.npy' % (dataset.name, self.name))
        quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), 'Initial')
        rootLogger.info(quality_desc)
        # Load network parameters - code borrowed from mnist lasagne example
        with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values, trainable=True)
        # reconstruction loss is just rms loss between input and reconstructed input
        reconstruction_loss = self.getReconstructionLossExpression(layers.get_output(self.network), self.t_target)
        # extent the network to do soft cluster assignments
        clustering_network = ClusteringLayer(self.encode_layer, dataset.getClusterCount(), cluster_centers, batch_size, self.encode_size)
        soft_assignments = layers.get_output(clustering_network)
        # k-means loss is the sum of distances from the cluster centers weighted by the soft assignments to the clusters
        kmeansLoss = self.getKMeansLoss(layers.get_output(self.encode_layer), soft_assignments, clustering_network.W, dataset.getClusterCount(), self.encode_size, batch_size)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        # total loss = reconstruction loss + lambda * kmeans loss
        weight_reconstruction = 1
        weight_kmeans = 0.1
        total_loss = weight_kmeans * kmeansLoss + weight_reconstruction * reconstruction_loss
        updates = lasagne.updates.nesterov_momentum(total_loss, params, learning_rate=0.01)
        trainKMeansWithAE = theano.function([self.t_input, self.t_target], total_loss, updates=updates)
        for epoch in range(epochs):
            error = 0
            total_batches = 0
            for batch in dataset.iterate_minibatches(self.input_type, batch_size, shuffle=True):
                inputs, targets = batch
                error += trainKMeansWithAE(inputs, targets)
                total_batches += 1
            # For every 10th epoch, update the cluster centers and print the clustering accuracy and nmi - for checking if the network
            # is actually doing something meaningful - the labels are never used for training
            if (epoch + 1) % 10 == 0:
                for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                    Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
                quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches))
                rootLogger.info(quality_desc)
            else:
                # Just print the training loss
                rootLogger.info("%-30s     %8s     %8s" % ("%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches), "", ""))
            if self.shouldStopNow:
            	break

        # Save the inputs in latent space and the network parameters
        for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
            Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
        np.save('saved_params/%s/pc_km_z_%s.npy' % (dataset.name, self.name), Z)
        np.savez('saved_params/%s/pc_km_m_%s.npz' % (dataset.name, self.name),
                 *lasagne.layers.get_all_param_values(self.network, trainable=True))


    def getKMeansLoss(self, latent_space_expression, soft_assignments, t_cluster_centers, num_clusters, latent_space_dim, num_samples, soft_loss=False):
        # Kmeans loss = weighted sum of latent space representation of inputs from the cluster centers
        z = latent_space_expression.reshape((num_samples, 1, latent_space_dim))
        z = T.tile(z, (1, num_clusters, 1))
        u = t_cluster_centers.reshape((1, num_clusters, latent_space_dim))
        u = T.tile(u, (num_samples, 1, 1))
        distances = (z - u).norm(2, axis=2).reshape((num_samples, num_clusters))
        if soft_loss:
            weighted_distances = distances * soft_assignments
            loss = weighted_distances.sum(axis=1).mean()
        else:
            loss = distances.min(axis=1).mean()
        return loss

    def networkToStr(self):
        # Utility method for printing the network structure in a shortened form
        layers = lasagne.layers.get_all_layers(self.network)
        result = ''
        for layer in layers:
            t = type(layer)
            if t is lasagne.layers.input.InputLayer:
                pass
            else:
                result += ' ' + layer.name
        return result.strip()


class NetworkBuilder(object):
    '''
    Class that handles parsing the architecture dictionary and creating an autoencoder out of it
    '''

    def __init__(self, network_description):
        '''
        :param network_description: python dictionary specifying the autoencoder architecture
        '''
        # Populate the missing values in the dictionary with defaults, also add the missing decoder part
        # of the autoencoder which is missing in the dictionary
        self.network_description = self.populateMissingDescriptions(network_description)
        # Create theano variables for input and output - would be of different types for simple and convolutional autoencoders
        if self.network_description['network_type'] == 'CAE':
            self.t_input = T.tensor4('input_var')
            self.t_target = T.tensor4('target_var')
            self.input_type = "IMAGE"
        else:
            self.t_input = T.matrix('input_var')
            self.t_target = T.matrix('target_var')
            self.input_type = "FLAT"
        self.network_type = self.network_description['network_type']
        self.batch_norm = bool(self.network_description["use_batch_norm"])
        self.layer_list = []

    def getBatchSize(self):
        return self.network_description["batch_size"]

    def getInputAndTargetVars(self):
        return self.t_input, self.t_target

    def getInputType(self):
        return self.input_type

    def buildNetwork(self):
        '''
        :return: Lasagne autoencoder network based on the network decription dictionary
        '''
        network = None
        for layer in self.network_description['layers']:
            network = self.processLayer(network, layer)
        return network

    def getEncodeLayerAndSize(self):
        '''
        :return: The encode layer - layer between encoder and decoder (bottleneck)
        '''
        return self.encode_layer, self.encode_size

    def populateDecoder(self, encode_layers):
        '''
        Creates a specification for the mirror of encode layers - which completes the autoencoder specification
        '''
        decode_layers = []
        for i, layer in reversed(list(enumerate(encode_layers))):
            if (layer["type"] == "MaxPool2D*"):
                # Inverse max pool doesn't upscale the input, but does reverse of what happened when maxpool
                # operation was performed
                decode_layers.append({
                    "type": "InverseMaxPool2D",
                    "layer_index": i,
                    'filter_size': layer['filter_size']
                })
            elif (layer["type"] == "MaxPool2D"):
                # Unpool just upscales the input back
                decode_layers.append({
                    "type": "Unpool2D",
                    'filter_size': layer['filter_size']
                })
            elif (layer["type"] == "Conv2D"):
                # Inverse convolution = deconvolution
                decode_layers.append({
                    'type': 'Deconv2D',
                    'conv_mode': layer['conv_mode'],
                    'non_linearity': layer['non_linearity'],
                    'filter_size': layer['filter_size'],
                    'num_filters': encode_layers[i - 1]['output_shape'][0]
                })
            elif (layer["type"] == "Dense" and not layer["is_encode"]):
                # Inverse of dense layers is just a dense layer, though we dont create an inverse layer corresponding to bottleneck layer
                decode_layers.append({
                    'type': 'Dense',
                    'num_units': encode_layers[i]['output_shape'][2],
                    'non_linearity': encode_layers[i]['non_linearity']
                })
                # if the layer following the dense layer is one of these, we need to reshape the output
                if (encode_layers[i - 1]['type'] in ("Conv2D", "MaxPool2D", "MaxPool2D*")):
                    decode_layers.append({
                        "type": "Reshape",
                        "output_shape": encode_layers[i - 1]['output_shape']
                    })
                if  (i+1<len(encode_layers) and encode_layers[i+1]['type']=="Concat"):
                    decode_layers[-1]['num_leading_axes']=0


        encode_layers.extend(decode_layers)

    def populateShapes(self, layers):
        # Fills the dictionary with shape information corresponding to each layer, which will be used in creating the decode layers
        last_layer_dimensions = layers[0]['output_shape']
        for layer in layers[1:]:
            if (layer['type'] == 'MaxPool2D' or layer['type'] == 'MaxPool2D*'):
                layer['output_shape'] = [last_layer_dimensions[0], last_layer_dimensions[1] / layer['filter_size'][0],
                                         last_layer_dimensions[2] / layer['filter_size'][1]]
            elif (layer['type'] == 'Conv2D'):
                multiplier = 1
                if (layer['conv_mode'] == "same"):
                    multiplier = 0
                layer['output_shape'] = [layer['num_filters'],
                                         last_layer_dimensions[1] - (layer['filter_size'][0] - 1) * multiplier,
                                         last_layer_dimensions[2] - (layer['filter_size'][1] - 1) * multiplier]
            elif (layer['type'] == 'Dense'):
                layer['output_shape'] = [1, 1, layer['num_units']]
            last_layer_dimensions = layer['output_shape']

    def populateMissingDescriptions(self, network_description):
        # Complete the architecture dictionary by filling in default values and populating description for decoder
        if 'network_type' not in network_description:
            if (network_description['name'].split('_')[0].split('-')[0] == 'fc'):
                network_description['network_type'] = 'AE'
            else:
                network_description['network_type'] = 'CAE'
        for layer in network_description['layers']:
            if 'conv_mode' not in layer:
                layer['conv_mode'] = 'valid'
            layer['is_encode'] = False
        network_description['layers'][-1]['is_encode'] = True
        if 'output_non_linearity' not in network_description:
            network_description['output_non_linearity'] = network_description['layers'][1]['non_linearity']
        self.populateShapes(network_description['layers'])
        self.populateDecoder(network_description['layers'])
        print(network_description['layers'])
        if 'use_batch_norm' not in network_description:
            network_description['use_batch_norm'] = False
        for layer in network_description['layers']:
            if 'is_encode' not in layer:
                layer['is_encode'] = False
            layer['is_output'] = False
        network_description['layers'][-1]['is_output'] = True
        network_description['layers'][-1]['non_linearity'] = network_description['output_non_linearity']
        return network_description
        
    def getInitializationFct(self):
		
		return lasagne.init.GlorotUniform()

    def processLayer(self, network, layer_definition):
        
        '''
        Create a lasagne layer corresponding to the "layer definition"
        '''
        print(layer_definition,"=====\n")
        ilayers=[] # will later be filled by the flattened layers in case of a concatlayer
        if (layer_definition["type"] == "Input"):
            if self.network_type == 'CAE':
                network = lasagne.layers.InputLayer(shape=tuple([None] + layer_definition['output_shape']), input_var=self.t_input)
            elif self.network_type == 'AE':
                network = lasagne.layers.InputLayer(shape=(None, layer_definition['output_shape'][2]), input_var=self.t_input)
        elif (layer_definition['type'] == 'Dense'):
            network = lasagne.layers.DenseLayer(network, num_units=layer_definition['num_units'], nonlinearity=self.getNonLinearity(layer_definition['non_linearity']), name=self.getLayerName(layer_definition),W=self.getInitializationFct())
        elif (layer_definition['type'] == 'Conv2D'):
            network = lasagne.layers.Conv2DLayer(network, num_filters=layer_definition['num_filters'], filter_size=tuple(layer_definition["filter_size"]), pad=layer_definition['conv_mode'], nonlinearity=self.getNonLinearity(layer_definition['non_linearity']), name=self.getLayerName(layer_definition),W=self.getInitializationFct())
        elif (layer_definition['type'] == 'MaxPool2D' or layer_definition['type'] == 'MaxPool2D*'):
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=tuple(layer_definition["filter_size"]), name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'InverseMaxPool2D'):
            network = lasagne.layers.InverseLayer(network, self.layer_list[layer_definition['layer_index']], name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'Unpool2D'):
            network = Unpool2DLayer(network, tuple(layer_definition['filter_size']), name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'Reshape'):
            network = lasagne.layers.ReshapeLayer(network, shape=tuple([-1] + layer_definition["output_shape"]), name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'Deconv2D'):
            network = lasagne.layers.Deconv2DLayer(network, num_filters=layer_definition['num_filters'], filter_size=tuple(layer_definition['filter_size']), crop=layer_definition['conv_mode'], nonlinearity=self.getNonLinearity(layer_definition['non_linearity']), name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'Concat'):
            for ilayer in layer_definition['input_layers_index']:
                network = lasagne.layers.flatten(self.layer_list[ilayer],outdim=1, name='fl')
                ilayers.append(network)
            network = lasagne.layers.ConcatLayer(ilayers, axis=0)
        self.layer_list.append(network)
        # Batch normalization on all convolutional layers except if at output
        if (self.batch_norm and (not layer_definition["is_output"]) and layer_definition['type'] in ("Conv2D", "Deconv2D")):
            network = batch_norm(network)
        # Save the encode layer separately
        if (layer_definition['is_encode']):
            self.encode_layer = lasagne.layers.flatten(network, name='fl')
            self.encode_size = layer_definition['output_shape'][0] * layer_definition['output_shape'][1] * layer_definition['output_shape'][2]
        return network
	
    def getLayerName(self, layer_definition):
        '''
        Utility method to name layers
        '''
        if (layer_definition['type'] == 'Dense'):
            return 'fc[{}]'.format(layer_definition['num_units'])
        elif (layer_definition['type'] == 'Conv2D'):
            return '{}[{}]'.format(layer_definition['num_filters'],
                                   'x'.join([str(fs) for fs in layer_definition['filter_size']]))
        elif (layer_definition['type'] == 'MaxPool2D' or layer_definition['type'] == 'MaxPool2D*'):
            return 'max[{}]'.format('x'.join([str(fs) for fs in layer_definition['filter_size']]))
        elif (layer_definition['type'] == 'InverseMaxPool2D'):
            return 'ups*[{}]'.format('x'.join([str(fs) for fs in layer_definition['filter_size']]))
        elif (layer_definition['type'] == 'Unpool2D'):
            return 'ups[{}]'.format(
                str(layer_definition['filter_size'][0]) + 'x' + str(layer_definition['filter_size'][1]))
        elif (layer_definition['type'] == 'Deconv2D'):
            return '{}[{}]'.format(layer_definition['num_filters'],
                                   'x'.join([str(fs) for fs in layer_definition['filter_size']]))
        elif (layer_definition['type'] == 'Reshape'):
            return "rsh"

    def getNonLinearity(self, non_linearity):
        return {
            'rectify': lasagne.nonlinearities.rectify,
            'linear': lasagne.nonlinearities.linear,
            'elu': lasagne.nonlinearities.elu
        }[non_linearity]
