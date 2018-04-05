import numpy as np
from TATi_neural_newtork import *
################### Prepare data #########################

X = np.asarray([[ 1.1892932,  -0.8064292 ],
 [ 2.34007037,  0.6482577 ],
 [-0.65681784,  2.18717645],
 [ 1.16332951 , 0.41719583],
 [ 1.7709225   ,2.25201178],
 [ 2.99424182,  2.84070527],
 [ 4.59369951,  3.16590588],
 [ 3.82765198,  3.14413603],
 [ 3.24654334,  4.10300203],
 [ 5.27317655,  2.75455839]])

Y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

####################################################

XY={}
XY['X'] = X
XY['Y'] = Y

neural_net = NeuralNetwork(XY) #optimize_for_initial_condition = False)
theta = neural_net.nn_parameters

print("The simplest gradient descent to test the force calls.")

nrsteps = 10
theta_n = np.zeros((nrsteps, theta.shape[0]))
f_n = np.zeros((nrsteps, theta.shape[0]))

loss_n = np.zeros(nrsteps)

print("Perform "+repr(nrsteps)+" steps.")

for n in range(nrsteps):

    f = np.copy(neural_net.force(theta))
    theta  = theta  +  0.01 * f

    f_n[n,:] = f
    theta_n[n,:] = np.copy(theta)
    loss_n[n] = neural_net.loss(theta)

print("################################")
print("Final values of the weights are:")
print(theta_n[-1])
print("################################")

predicted_classes = neural_net.predict(theta_n[-1])

print("################################")
print("Prediction: ")
print(predicted_classes)
print("Accuracy: ")
acc = neural_net.accuracy(theta_n[-1], print_score=False)
print(acc)
print("################################")
print("Loss at the first step:  ")
print(loss_n[0])
print("Loss at the last step:  ")
print(loss_n[-1])
#
#print(theta_n)
#print(f_n)
