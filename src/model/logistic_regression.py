# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from random import shuffle

from util.activation_functions import Activation
from model.classifier import Classifier
from logistic_layer import LogisticLayer

from util.loss_functions import DifferentError
from util.loss_functions import AbsoluteError
from util.loss_functions import BinaryCrossEntropyError
from util.loss_functions import CrossEntropyError
from util.loss_functions import SumSquaredError
from util.loss_functions import MeanSquaredError

# plotting libaries
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

    def __init__(self, train, valid, test, learningRate=1e-3, epochs=30):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # TODO multilayer perceptron
        nIn, nOut = self.trainingSet.input.shape[1], 1
        self.layer = LogisticLayer(nIn, nOut, weights=None, activation="sigmoid", isClassifierLayer=True)

    def train(self, verbose=True):
        """
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

        learned = False
        it = 0
        totE = 0        # total error
        errHist = []    # error history

        # Instead of all labels in one array, we make a matrix of [1xL] where L is the number of labels
        labels = np.matrix(np.column_stack((self.trainingSet.label,)))
        # add bias (column of ones)
        inputs = np.matrix(np.append(np.ones((self.trainingSet.input.shape[0], 1)), self.trainingSet.input, axis=1))

        while not learned:
            # shuffle the data in a convenient way
            data = zip(inputs, labels)
            shuffle(data)

            for input, label in data:
                # forward pass
                output = self._forwardPass(input)
                # compute the error
                totE += abs(loss.calculateError(label, output))
                # backward pass
                self._backwardPass(label, output)
                # update weights
                self._updateWeights()

            # update local variables
            errHist.append(totE)
            it += 1

            # compute error difference for logging purposes
            deltaE = 0 if it < 2 else abs(errHist[-1] - errHist[-2])

            if verbose:
                logging.info("Epoch: %i; Error: %i; ΔE: %f", it, totE, deltaE)

            # convergence criteria
            learned = (totE == 0 or it >= self.epochs)
            totE = 0

        self._plotResults(errHist)

    def _forwardPass(self, input):
        return self.layer.forward(input)

    def _backwardPass(self, label, output):
        self.layer.computeDerivative(-(label-output), np.matrix([[1]]))

    def _updateWeights(self):
        # TODO implement other policies for learningRates
        self.layer.updateWeights(self.learningRate)

        # classify as single instance
    def classify(self, testInstance):
        """
        testInstance : list of floats

        Returns: True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.layer.forward(np.array([np.concatenate((np.array([1]),testInstance))]))[0] > 0.5

        # evaluate the whole dataset
    def evaluate(self, test=None):
        """
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns: List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def _plotResults(self, errorHistory):
        # 'python-tk package not found?' => apt-get install python-tk

        # Make errorHistory a list of values instead of matrices
        errorHistory = [e.item(0) for e in errorHistory]
        plt.plot(errorHistory, '-b',  label='Error cost history over training set')
        plt.ylabel('Error cost')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        # enable this to show graph
        plt.show()
        pass
