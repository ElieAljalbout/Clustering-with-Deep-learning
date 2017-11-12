'''
Created on Jul 25, 2017
'''

from lasagne import layers
import theano
import theano.tensor as T


class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    Layer borrowed from: https://swarbrickjones.wordpress.com/2015/04/29/convolutional-autoencoders-in-pythontheanolasagne/
    """

    def __init__(self, incoming, ds, **kwargs):
        super(Unpool2DLayer, self).__init__(incoming, **kwargs)
        self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]
        return tuple(output_shape)

    def get_output_for(self, incoming, **kwargs):
        '''
        Just repeats the input element the upscaled image
        '''
        ds = self.ds
        return incoming.repeat(ds[0], axis=2).repeat(ds[1], axis=3)


class ClusteringLayer(layers.Layer):
    '''
    This layer gives soft assignments for the clusters based on distance from k-means based
    cluster centers. The weights of the layers are the cluster centers so that they can be learnt
    while optimizing for loss
    '''

    def __init__(self, incoming, num_clusters, initial_clusters, num_samples, latent_space_dim, **kwargs):
        super(ClusteringLayer, self).__init__(incoming, **kwargs)
        self.num_clusters = num_clusters
        self.W = self.add_param(theano.shared(initial_clusters), initial_clusters.shape, 'W')
        self.num_samples = num_samples
        self.latent_space_dim = latent_space_dim

    def get_output_shape_for(self, input_shape):
        '''
        Output shape is number of inputs x number of cluster, i.e for each input soft assignments
        corresponding to all clusters
        '''
        return (input_shape[0], self.num_clusters)

    def get_output_for(self, incoming, **kwargs):
        return getSoftAssignments(incoming, self.W, self.num_clusters, self.latent_space_dim, self.num_samples)


def getSoftAssignments(latent_space, cluster_centers, num_clusters, latent_space_dim, num_samples):
    '''
    Returns cluster membership distribution for each sample
    :param latent_space: latent space representation of inputs
    :param cluster_centers: the coordinates of cluster centers in latent space
    :param num_clusters: total number of clusters
    :param latent_space_dim: dimensionality of latent space
    :param num_samples: total number of input samples
    :return: soft assigment based on the equation qij = (1+|zi - uj|^2)^(-1)/sum_j'((1+|zi - uj'|^2)^(-1))
    '''
    z_expanded = latent_space.reshape((num_samples, 1, latent_space_dim))
    z_expanded = T.tile(z_expanded, (1, num_clusters, 1))
    u_expanded = T.tile(cluster_centers, (num_samples, 1, 1))

    distances_from_cluster_centers = (z_expanded - u_expanded).norm(2, axis=2)
    qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
    qij_numerator = 1 / qij_numerator
    normalizer_q = qij_numerator.sum(axis=1).reshape((num_samples, 1))

    return qij_numerator / normalizer_q
