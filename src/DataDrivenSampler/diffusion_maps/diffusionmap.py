"""Diffusion map"""

# Author: Zofia
# License:


import numpy as np

import math
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as SLA

import sklearn.neighbors as neigh_search
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sps

from numpy import linalg as LA

# compute sparse matrices from a given trajectory


def compute_kernel(X, epsilon):
    """ Computes a sparse, gaussian kernel between points on a trajectory

    :param X: vector with trajectories
    :param epsilon: Determines the distance cutoff in nearest neighbor search
    :return: sparse kernel matrix
    """

    m = np.shape(X)[0]

    cutoff = 4.0*np.sqrt(2*epsilon)


    #calling nearest neighbor search class: returning a (sparse) distance matrix
    albero = neigh_search.radius_neighbors_graph(X, radius = cutoff,mode='distance', p=2, include_self=None)


    # computing the diffusion kernel value at the non zero matrix entries
    diffusion_kernel = np.exp(-(np.array(albero.data)**2)/(epsilon))

    # build sparse matrix for diffusion kernel
    kernel = sps.csr_matrix((diffusion_kernel, albero.indices, albero.indptr), dtype = float, shape=(m,m))
    kernel = kernel + sps.identity(m)  # accounting for diagonal elements

    return kernel


def compute_VanillaDiffusionMap(kernel, X):
    """ Rescales the original diffusion map by looking at the sparse kernel's diagonal.

    :param kernel: kernel matrix
    :param X: trajectory vector
    :return: rescaled kernel matrix
    """
    alpha = 0.5
    m = np.shape(X)[0]

    D = sps.csr_matrix.sum(kernel, axis=0)
    Dalpha = sps.spdiags(np.power(D,-alpha), 0, m, m)
    kernel = Dalpha * kernel * Dalpha

    # normalize matrix
    D = sps.csr_matrix.sum(kernel, axis=0)
    Dalpha = sps.spdiags(np.power(D,-1), 0, m, m)
    kernel = Dalpha * kernel

    return kernel


def compute_target_distribution(number_of_steps, beta, loss):
    """ Computes the target Boltzmann distribution

    :param number_of_steps: number of trajectory steps
    :param beta: inverse temperature parameter
    :param loss: potential or loss function along trajectory
    :return: target distribution (vector with number_of_steps)
    """
    qTargetDistribution = np.zeros(number_of_steps)

    for i in range(0, number_of_steps):
        qTargetDistribution[i] = np.exp(-(loss[i] * beta))
        #print("#"+str(i)+": Loss "+str(loss[i])+" beta "+str(beta)+": qTarget "+str(qTargetDistribution[i]))

    return qTargetDistribution


def compute_TMDMap(X, epsilon, qTargetDistribution):
    """ Computes the TM diffusion map variant.

    :param X: trajectory vector
    :param epsilon: cutoff parameter for constructing kernel matrix
    :param qTargetDistribution: target Boltzmann distribution
    :return:
    """

    #print('Unweighting according to temperature '+repr(temperature))
    m = np.shape(X)[0]

    kernel = compute_kernel(X, epsilon)

    qEmp=kernel.sum(axis=1)

    # reweight matrix by ratio of target over empirical distribution
    weights = np.zeros(m)
    for i in range(0,len(X)):
        weights[i] = np.sqrt(qTargetDistribution[i]) / qEmp[i]
    #print("Weights for targetted mean: "+str(weights[0:10]))
    D = sps.spdiags(weights, 0, m, m)
    Ktilde =  kernel * D

    # normalize matrix
    Dalpha = sps.csr_matrix.sum(Ktilde, axis=0)
    Dtilde = sps.spdiags(np.power(Dalpha,-1), 0, m, m)

    L = Dtilde * Ktilde

    return L, qEmp


def pdist2(x,y):
    """ Computes the euclidian distance for two vectors.

    :param x: first vector
    :param y: second vector
    :return: euclidian distance between the two vectors
    """
    v=np.sqrt(((x-y)**2).sum())
    return v


def get_landmarks(data, K, q, V, energies):
    """

    :param data: trajectory vector
    :param K: number of landmark points to obtain
    :param q: empirial distribution
    :param V: eigenvector (of diffusion map) to use
    :param energies: loss or potential function
    :return:
    """

    m = float(q.size)
    #q=np.array(q)

    delta = 100/m*(max(V) - min(V))
    deltaMax=2*delta
    levels = np.linspace(min(V), max(V), num=K)

    lb = 1

    landmarks=np.zeros(K)
    emptyLevelSet=0

    for k in range(K-1, -1, -1):


            levelsetLength=0

            # we want to identify indices in V which are delta close to the levels
            #---o----o----o----o----o---
            #  *o** *o*  *o*  *o*   o
            # if there are no indices in the delta distance, increase the delta distance

            while levelsetLength==0:

                levelset = np.where(np.abs(V - levels[k]) < delta)
                levelset=levelset[0]
                levelsetLength=len(levelset)

                delta=delta*1.001

                if delta>deltaMax:
                    levelset=range(0, len(V))

            data_level = data[levelset,:]

            if k==K-1:

                idx = np.argmin(energies[levelset] / m)
                landmarks[k]= levelset[idx]

            else:

                idx = np.argmin(energies[levelset] / m)
                qtmp= energies[levelset] / m

                # compute the distance to the last landmark
                dist_to_last=np.zeros(data_level.shape[0])
                for i in range(0,data_level.shape[0]):
                    dist_to_last[int(i)] = pdist2(data[int(landmarks[k+1]),:], data_level[int(i)])
                dtmp=np.array(dist_to_last.reshape(qtmp.shape))

                v=qtmp  - lb*dtmp

                idx = np.argmax(v);

                landmarks[k]= levelset[idx]

    return landmarks.astype(int)


def sort_landmarks(data, landmarks):
    """ Sort the landmark points according to ball_tree variant of nearest neighbor search.

    :param data: trajectory
    :param landmarks:  landmarks
    :return: sorted landmarks
    """

    X=data[landmarks,:]

    #if len(X.shape)>2:
    #    X=reshapeData(X)


    nbrs = NearestNeighbors(n_neighbors=len(landmarks), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return landmarks[indices[0]]


def get_levelsets(data, K, q, V1):
    """ Calculates the K level set points given a trajectory.

    :param data: trajectory vector
    :param K: number of level sets to calculate
    :param q: empirial distribution
    :param V1: eigenvector
    :return: level set values
    """

    m = float(q.size)
    #q=np.array(q)

    delta = 100/m*(max(V1)-min(V1))
    deltaMax=2*delta
    levels = np.linspace(min(V1),max(V1),num=K)

    lb = 1

    landmarks=np.zeros(K)
    emptyLevelSet=0

    levelsets=list()

    for k in range(K-1, -1, -1):

            levelsetLength=0

            # we want to identify idices in V1 which are delta close to the levels
            #---o----o----o----o----o---
            #  *o** *o*  *o*  *o*   o
            # if there are no indeces in the delta distance, increase the delta distance

            while levelsetLength==0:

                levelset_k = np.where(np.abs(V1 - levels[k]) < delta)
                levelsetLength=len(levelset_k)
                #print levelsetLength

                delta=delta*1.001

                if delta>deltaMax:
                    levelset_k=range(0,len(V1))
                    #print("In get_landmarks: Levelset chosen as V1")

                levelsets.append(levelset_k[0])

    return levelsets, levels


def computeFreeEnergyAtEveryPoint(X, V1, width, qTarget, qEmp, method='weighted'):
    """

    :param X: trajectory
    :param V1: eigenvector
    :param width: binning width
    :param qTarget: target distribution
    :param qEmp: empirial distribution
    :param method: either weighted or raw (use weighted if sampled trajectories do
           not follow the target distribution and need reweighting in order to do so)
    :return: free energies calculated at all trajectory points
    """
    if(method=='weighted'):
        weight, Ntilde = compute_weight_target_distribution(qTarget, qEmp)

    freeEnergy=np.zeros(len(X))
    h=np.zeros(len(X))

    for k in range(0,len(X)):
        levelset = get_levelset_onePoint(k, width, V1)
        #print(len(levelset))

        # simple histogram
        if(method == 'raw'):
            h[k] = np.sum(len(levelset))
        elif(method=='weighted'):
            # unbiased histogram- we can use weights!!
            h[k] = np.sum(weight[levelset])

    for k in range(0,len(X)):
        if(h[k] == 0):
            freeEnergy[k] = 0 # if the bin is empty set 0
        else:
            if(method == 'raw'):
                freeEnergy[k] = -np.log(h[k]/sum(h))
            elif(method=='weighted'):
                freeEnergy[k] =  -np.log( h[k]/ Ntilde)

    if(method=='weighted'):
        return freeEnergy#, weight, Ntilde
    else:
        return freeEnergy


def get_levelset_onePoint(idx, width, V1):
    """ Return the levelset at a single point

    :param idx: index of the desired point
    :param width: width of the level set
    :param V1: eigenvector
    :return:
    """
    #deltaMax=2*width

    #print(np.abs(V1 - V1[idx]))

    if V1[idx] == 0:
        tmp=( np.where(np.abs(V1 - V1[idx]) < width))
    else:
        tmp=( np.where(np.abs(V1 - V1[idx])/np.abs(V1[idx]) < width))

    levelset= tmp[0]

    return np.asarray(levelset)


def compute_weight_target_distribution(target_distribution, qImportanceSampling):
    """ Reweight the target distribution by another as obtained by importance sampling.

    :param target_distribution: target distribution vector
    :param qImportanceSampling: distribution use for reweighting
    """

    ModNr=1

    weight=np.zeros(len(qImportanceSampling))

    for count in range(len(qImportanceSampling)):
        weight[count]=target_distribution[count]/qImportanceSampling[count]

    Ntilde=np.sum(weight)

    return weight, Ntilde