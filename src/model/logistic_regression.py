# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from util.loss_functions import DifferentError
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
        DE = DifferentError()
        AE = AbsoluteError()
        BCE = BinaryCrossEntropyError()

        # use loss to choose error function
        loss = DE
        # See
        # https://ilias.studium.kit.edu/goto.php?target=file_701288_download&client_id=produktiv
        # for batch version slide - we think it's wrong somehow
        while(epoch < self.epochs):
            gradient = np.zeros(len(self.weight))
            numE = 0
            for input, target in zip(self.trainingSet.input, self.trainingSet.label):
                output = self.fire(input);
                error = loss.calculateError(target, output)

                #gradient += - (output - target)
                if isinstance(loss, DifferentError):
                    gradient += - error * input
                elif isinstance(loss, AbsoluteError):
                    gradient += - error * input # TODO change
                elif isinstance(loss, BinaryCrossEntropyError):
                    gradient = gradient + np.dot((self.fire(data) - label), data)
                else:
                    gradient += - error * input

                numE += abs(error)

            self.updateWeights(gradient)

            if verbose:
                logging.info("Epoch: %i; Error: %i", epoch, numE)

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
        return self.fire(testInstance) > 0.5
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
        #self.weight = [(w - self.learningRate * g) for w, g in zip(self.weight, grad)]
        self.weight += - self.learningRate * grad
        pass

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
