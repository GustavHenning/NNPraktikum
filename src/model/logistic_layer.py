
import time

import numpy as np

from util.activation_functions import Activation
#from model.layer import Layer


class LogisticLayer():
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, isInputLayer=False,
                 activation='sigmoid', isClassifierLayer=False):
        ##
        ## TODO it's actually supposed to be softmax activation !!! TODO
        ##

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationPrime = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        # You can have better initialization here

        rns = np.random.RandomState(int(time.time()))
        self.weights = rns.uniform(size=(nOut, nIn + (1 if isInputLayer else 0)))-0.5
        #print(str(self.weights.shape))


        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        self.input = input
        # [x * w] => [S] => y
        #print("x: " + str(self.input.shape) + " * w:" + str(self.weights.shape) + " = ")
        #print(str(np.dot(self.weights, np.array(self.input)).shape))
        self.output = self.activation(np.dot(self.weights, np.array(self.input)))
        return self.output

    def computeDerivative(self, label, loss, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        label:
            used by classification layers.
        loss:
            used by classification layers.
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        if self.isClassifierLayer:
        #    print("y^: " + str(self.activationPrime(self.output).shape) +
        #    " *  [l: " + str(np.array(label)) +
        #    " - y: " + str(self.output.shape) + " ]")
        #    print(" = " + str(np.multiply(self.activationPrime(self.output), np.array(label - self.output)).shape))
        # label - self.output
            self.delta = np.multiply(self.activationPrime(self.output), loss.calculateError(np.array(label), np.array(self.output)))
        else:
        #    print("Unimplemented for multiple layers")
        #    print("nD: " + str(nextDerivatives.shape) +
        #    "nW: " + str(nextWeights.shape) +
        #    " * y^: " + str(self.activationPrime(self.output).shape))
            self.delta = np.dot(nextDerivatives, nextWeights) * self.activationPrime(self.output)
        return self.delta

    def updateWeights(self, learningRate=0.01):
        """
        Update the weights of the layer
        """
        #print("w: " + str(self.weights.shape) + " = d: " + str(self.delta.shape) + " * i: " + str(self.input.shape))
        #print(np.self.weights)
        #print(str(self.delta[:,np.newaxis]))
        self.weights += learningRate * self.input * self.delta[:,np.newaxis]
