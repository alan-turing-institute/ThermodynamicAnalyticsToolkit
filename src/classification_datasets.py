import math
import numpy as np

from dataset import dataset

class classification_datasets:
    ''' This class encapsulates all datasets from the TensorFlow playground
    for classification tasks.
    '''

    TWOCIRCLES=0
    SQUARES=1
    TWOCLUSTERS=2
    SPIRAL=3
    datasettypes = {
        TWOCIRCLES: "generator_twocircles",
        SQUARES: "generator_squares",
        TWOCLUSTERS: "generator_twoclusters",
        SPIRAL: "generator_spiral" }


    xs = []
    ys = []
    r = 5
    
    def generate(self, dimension, noise, data_type=SPIRAL):
        '''
        Generates the input data where data_type decides which
        type to generate. All data resides in the domain [-r,r]^2.
        '''
        # clear return instances
        self.xs[:] = []
        self.ys[:] = []
        # call dataset generating function
        if data_type == self.TWOCIRCLES:
            self.generate_twocircles(dimension, noise)
        elif data_type == self.SQUARES:
            self.generate_squares(dimension, noise)
        elif data_type == self.TWOCLUSTERS:
            self.generate_twoclusters(dimension, noise)
        elif data_type == self.SPIRAL:
            self.generate_spiral(dimension, noise)
        else:
            print("Unknown input data type desired.")
        return dataset(self.xs, self.ys)

    def generate_twocircles(self, dimension, noise):
        ''' Generates two circular distributions with the same
            center but different radii, i.e. they are mostly not
            overlapping.
        '''
        for label in [1,-1]:
            for i in range(int(dimension/2)):
                if label == 1:
                    radius = np.random.uniform(0,self.r*0.5)
                else:
                    radius = np.random.uniform(self.r*0.7, self.r)
                angle = np.random.uniform(0,2*math.pi)
                coords = [radius * math.sin(angle), radius * math.cos(angle)]
                noisecoords = np.random.uniform(-self.r,self.r,2)*noise
                norm = (coords[0]+noisecoords[0])*(coords[0]+noisecoords[0])+(coords[1]+noisecoords[1])*(coords[1]+noisecoords[1])
                self.xs.append(coords)
                self.ys.append([1, 0] if (norm < self.r*self.r*.25) else [0, 1])
                #print(str(returndata[-1])+" with norm "+str(norm)+" and radius "+str(radius)+": "+str(labels[-1]))
                
    def generate_squares(self, dimension, noise):
        ''' Generates distribution in each of the four quadrants
            of the two-dimensional domain.
        '''
        for i in range(dimension):
            coords = np.random.uniform(-self.r,self.r,2)
            padding = .3
            coords[0] += padding * (1 if (coords[0] > 0) else -1)
            coords[1] += padding * (1 if (coords[1] > 0) else -1)
            noisecoords = np.random.uniform(-self.r,self.r,2)*noise
            self.xs.append(coords)
            self.ys.append([1, 0] if ((coords[0]+noisecoords[0])*(coords[1]+noisecoords[1]) >= 0) else [0, 1])

    def generate_twoclusters(self, dimension, noise):
        ''' Generates two normal distribution point clouds centered at
            [2,2] and [-2,-2].
        '''
        variance = 0.5+noise*(3.5*2)
        signs=[1,-1]
        labels=[[1,0],[0,1]]
        for i in range(2):
            for j in range(int(dimension/2)):
                coords = np.random.normal(signs[i]*2,variance,2)
                self.xs.append(coords)
                self.ys.append(labels[i])

    def generate_spiral(self, dimension, noise):
        ''' Generates two spiral-shaped distributions each with a different
            label. This is a standard example for distributions that cannot
            be sensibly distinguished by a linear ansatz, i.e. just a single
            hidden layer.
        '''
        for deltaT in [0, math.pi]:
            for i in range(int(dimension/2)):
                radius = i/dimension*self.r
                t = 3.5 * i/dimension* 2*math.pi + deltaT
                coords = [radius*math.sin(t)+np.random.uniform(-1,1)*noise,
                          radius*math.cos(t)+np.random.uniform(-1,1)*noise]
                self.xs.append(coords)
                self.ys.append([1, 0] if (deltaT == 0) else [0, 1])
        