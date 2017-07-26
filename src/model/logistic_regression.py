# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

from util.loss_functions import DifferentError
from util.loss_functions import AbsoluteError
from util.loss_functions import BinaryCrossEntropyError
from util.loss_functions import CrossEntropyError
from util.loss_functions import SumSquaredError
from util.loss_functions import MeanSquaredError

from scipy.misc import derivative

import pdb
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=10, layerConf=[2,1]):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        self.layers = []
        # iteratve over layerConf and instansiate the layers
        # layerConf contains the size of the output of each layer
        # TODO: Probably different activations for hidden/output (softmax/sigmoid)
        for ind, val in enumerate(layerConf):
            #print(str(ind) + " " + str(val))
            # For the first layer, nIn == size of input, nOut defined by layerConf
            # Any other layer, size of input equal to that of the previous output.
            sizeIn = layerConf[ind - 1] if ind != 0 else self.trainingSet.input.shape[1]
            sizeOut = val
            activeStrings = ['sigmoid', 'softmax', 'tanh', #'linear'
            ];
            randomActive = activeStrings[np.random.randint(0, len(activeStrings))]
            # "sigmoid"
            actives = "sigmoid" if ind == len(layerConf) - 1 else randomActive
            # in, out, isInput, acitvationFunction, isClassifier
            #print(str(sizeIn) + " " + str(sizeOut))
            self.layers.append(LogisticLayer(sizeIn, sizeOut, ind == 0, actives))
            logging.info("Layer %i, size [%i → %i], %s", ind, sizeIn, sizeOut, actives)

        # set the last layer to be a classifier
        self.layers[-1].isClassifierLayer = True

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

        # Bias
        self.trainingSet.input = np.insert(self.testSet.input, 0, 1, axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1, axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        DE = DifferentError() # Working
        AE = AbsoluteError() # nope
        BCE = BinaryCrossEntropyError() # Working(?)
        CE = CrossEntropyError() # Not implemented
        SSE = SumSquaredError() # Working(?)
        MSE = MeanSquaredError() # Working(?)
        # ----------------------------------
        # use loss to choose error function
        # ----------------------------------
        loss = DE
        loss.errorString()
        # ----------------------------------
        if verbose:
            logging.info("LogRes using [%s] with %i epochs", loss.errorString, self.epochs)
        learned = False
        epoch = 1
        CLASSIFIER_LAYER = 0 # classifier layer is the first in the reverse list
        trainingCost = np.zeros(self.epochs)
        while not learned:
            totalError = 0
            for input, label in zip(self.trainingSet.input, self.trainingSet.label):

                ##
                ## Feed forward step - let the layers evaluate input in a cascading manner
                ##
                output = self.forward(input)
                #totalError -= loss.calculateError(label, output)

                ##
                ## Backpropagation step - From back to front, calculate nextDerivates
                ## and pass the weights to the next layer
                ##
                for ind, layer in enumerate(reversed(self.layers)):
                    if ind == CLASSIFIER_LAYER:
                        # The classifier layer. NOTE: Probably needs specific arguments
                        # depending on the error function!
                        totalError -= layer.computeDerivative(label, loss, None, None)
                    else:
                        # hidden layer: propagate backwards with the derivates and the weights
                        # The minus sign indicates going from right to left index-wise.
                        layer.computeDerivative(None, None, self.layers[-ind].delta, self.layers[-ind].weights)

                ##
                ## Update step
                ##
                for layer in self.layers:
                    layer.updateWeights(self.learningRate)

            trainingCost[epoch-1] = totalError

            if verbose:
                logging.info("Epoch: %i; Error: %f", epoch, totalError)

            if epoch >= self.epochs:
                learned = True
            epoch = epoch + 1

        # 'python-tk package not found?' => apt-get install python-tk
        plt.plot(trainingCost, '-b',  label='Training Set')
        plt.ylabel('Error cost')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        # enable this to show graph
        #plt.show()
        pass

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        temp = testInstance
        return self.forward(temp) > 0.5
        pass

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))


    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
