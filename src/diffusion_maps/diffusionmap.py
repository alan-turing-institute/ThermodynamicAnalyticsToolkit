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


def reshapeData(X):

    m, nrP, dim=X.shape
    Xreshaped=np.zeros((m,nrP*dim))
    for i in range(0,m):
        for j in range(0, nrP):
            Xreshaped[i,j*dim:(j+1)*dim]=X[i,j,:]
    return Xreshaped


# compute sparse matrices from a given trajectory


def compute_kernel(X, epsilon):

    m = np.shape(X)[0];

    cutoff = 4.0*np.sqrt(2*epsilon);


    #calling nearest neighbor search class: returning a (sparse) distance matrix
    albero = neigh_search.radius_neighbors_graph(X, radius = cutoff,mode='distance', p=2, include_self=None)


    # computing the diffusion kernel value at the non zero matrix entries
    diffusion_kernel = np.exp(-(np.array(albero.data)**2)/(epsilon))

    # build sparse matrix for diffusion kernel
    kernel = sps.csr_matrix((diffusion_kernel, albero.indices, albero.indptr), dtype = float, shape=(m,m))
    kernel = kernel + sps.identity(m)  # accounting for diagonal elements

    return kernel;

def compute_local_kernel(X, A, b, epsilon):

    #A = \beta^{-1}I and b = -\nabla U
    beta = 1.0/A

    m = np.shape(X)[0];

    cutoff = 4.0*np.sqrt(2*epsilon);

    #calling nearest neighbor search class: returning a (sparse) distance matrix
    albero = neigh_search.radius_neighbors_graph(X , radius = cutoff,mode='distance', p=2, include_self=None)

    # computing the diffusion kernel value at the non zero matrix entries
    diffusion_kernel = np.exp(- beta * (np.array(albero.data )**2)/(4.0*epsilon))

    # build sparse matrix for diffusion kernel
    kernel = sps.csr_matrix((diffusion_kernel, albero.indices, albero.indptr), dtype = float, shape=(m,m))
    kernel = kernel + sps.identity(m)  # accounting for diagonal elements

    return kernel;

def compute_P(kernel, X):


    alpha = 0.5;
    m = np.shape(X)[0];

    D = sps.csr_matrix.sum(kernel, axis=0);
    Dalpha = sps.spdiags(np.power(D,-alpha), 0, m, m)
    kernel = Dalpha * kernel * Dalpha;

    D = sps.csr_matrix.sum(kernel, axis=0);
    Dalpha = sps.spdiags(np.power(D,-1), 0, m, m)
    kernel = Dalpha * kernel;

    return kernel

def compute_unweighted_P( X, epsilon, sampler):

    print('Unweighting according to temperature '+repr(sampler.T))
    m = np.shape(X)[0];

    kernel = compute_kernel(X, epsilon)

    qEmp=kernel.sum(axis=1)

    target_distribution = np.zeros(m)
    weights = np.zeros(m)

    for i in range(0,len(X)):

        target_distribution[i] = np.exp( -  sampler.model.potential(X[i]) / sampler.T)
        weights[i] = np.sqrt(target_distribution[i]) /  qEmp[i]

    D = sps.spdiags(weights, 0, m, m)
    Ktilde =  kernel * D

    Dalpha = sps.csr_matrix.sum(Ktilde, axis=0);
    Dtilde = sps.spdiags(np.power(Dalpha,-1), 0, m, m)

    L = Dtilde * Ktilde

    return L, target_distribution, qEmp

def compute_local_unweighted_P( X, epsilon, sampler):

    print('Unweight according to temperature '+repr(sampler.T))
    m = np.shape(X)[0];

    b =np.zeros(X.shape)
    for i in range(len(b)):
        b[i,:] = sampler.model.force(X[i])

    A = sampler.T

    W = compute_local_kernel(X, A, b, epsilon)

    kernel = compute_kernel(X,  epsilon)
    qEmp=kernel.sum(axis=1)

    Dq = sps.spdiags(qEmp[0], 0, m, m)
    Ktilde =  W * Dq

    L_loc = sps.csr_matrix.sum(Ktilde, axis=0);
    L_loc = sps.spdiags(np.power(L_loc,-1), 0, m, m)
    L_loc = L_loc*W

    Ltmp = sps.csr_matrix.sum(Ktilde, axis=0);
    Ltmp = sps.spdiags(np.power(Ltmp,-1), 0, m, m)

    L = L_loc
    #L = (L_loc - Ltmp)/epsilon


    return L, qEmp



def pdist2(x,y):
    v=np.sqrt(((x-y)**2).sum())
    return v

def get_eigenvectors(data, nrEV, **kwargs):

    if ('DiffMapMatrix' in kwargs):
         P= kwargs['DiffMapMatrix']
    else:
         P,q = compute_P_and_q(data)
    lambdas, V = SLA.eigsh(P, k=nrEV)#, which='LM' )
    ix = lambdas.argsort()[::-1]
    return V[:,ix], lambdas[ix]




def get_landmarks(data, K, q, V1, potEn):

    m = float(q.size)
    #q=np.array(q)

    delta = 100/m*(max(V1)-min(V1))
    deltaMax=2*delta
    levels = np.linspace(min(V1),max(V1),num=K)

    lb = 1

    landmarks=np.zeros(K)
    emptyLevelSet=0

    # compute potential energy on the data points (usually available in the codes..)
    E=np.zeros(len(data))
    for n in range(0,len(data)):
        E[n] = potEn(data[n,:])

    for k in range(K-1, -1, -1):


            levelsetLength=0

            # we want to identify idices in V1 which are delta close to the levels
            #---o----o----o----o----o---
            #  *o** *o*  *o*  *o*   o
            # if there are no indeces in the delta distance, increase the delta distance

            while levelsetLength==0:

                levelset = np.where(np.abs(V1 - levels[k]) < delta)
                levelset=levelset[0]
                levelsetLength=len(levelset)
                #print levelsetLength

                delta=delta*1.001

                if delta>deltaMax:
                    levelset=range(0,len(V1))
                    #print("In get_landmarks: Levelset chosen as V1")

            data_level = data[levelset,:]

            if k==K-1:
                #idx = np.argmax(q[levelset]/m )
                idx = np.argmin(E[levelset]/m )
                landmarks[k]= levelset[idx]

            else:
                #idx = np.argmax(q[levelset]/m )
                #qtmp=q[levelset]/m
                idx = np.argmin(E[levelset]/m )
                qtmp=E[levelset]/m

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

    X=data[landmarks,:]

    #if len(X.shape)>2:
    #    X=reshapeData(X)


    nbrs = NearestNeighbors(n_neighbors=len(landmarks), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return landmarks[indices[0]]



def sort_landmarkedPoints(gridCV):

    dim=gridCV.shape[1]
    X=gridCV[:,0:dim+1]

    if len(X.shape)>2:
        X=reshapeData(X)


    nbrs = NearestNeighbors(n_neighbors=len(X), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return gridCV[indices[0], :]

def compute_eigenvectors(laplacian):

    #random_state = check_random_state(random_state)

    laplacian *= -1
    v0 = np.random.uniform(-1, 1, laplacian.shape[0])
    lambdas, diffusion_map = eigh(laplacian, k=n_components,
                                         sigma=1.0, which='LM',
                                         tol=eigen_tol, v0=v0)
    return diffusion_map.T[n_components::-1] # * dd



def get_levelsets(data, K, q, V1):

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
