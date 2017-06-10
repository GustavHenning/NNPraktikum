# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from util.loss_functions import AbsoluteError
from util.loss_functions import BinaryCrossEntropyError
from scipy.misc import derivative

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

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        epoch = 0
        AE = AbsoluteError()
        BCE = BinaryCrossEntropyError()
        # See
        # https://ilias.studium.kit.edu/goto.php?target=file_701288_download&client_id=produktiv
        # for batch version slide - we think it's wrong somehow
        while(epoch < self.epochs):
            grad = np.zeros(len(self.weight))
            # fire on all the inputs
            output = [self.fire(data) for data in self.trainingSet.input]
            # calculate the error over all outputs
            # We have to derivate this function somehow
            error = BCE.calculateError(self.trainingSet.label, output)

            if verbose:
                logging.info("Epoch: %i; Error: %i", epoch, error)


            for data in self.trainingSet.input:
                grad = [(g + error * x) for g, x in zip(grad, data)]

            if verbose:
                logging.info("Epoch: %i; Grad: %i", epoch, sum(grad))

            self.updateWeights(grad)
            epoch = epoch + 1

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

        # maybe w - self.learningRate? * g
    def updateWeights(self, grad):
        self.weight = [(w + self.learningRate * g) for w, g in zip(self.weight, grad)]
        pass

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
