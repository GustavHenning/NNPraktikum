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

    def __init__(self, train, valid, test, layerConf=[2,1], learningRate=1e-3, epochs=100):

        np.random.seed(1)
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
            actives = "tanh" #if ind == len(layerConf) - 1 else "sigmoid"
            # in, out, isInput, acitvationFunction, isClassifier
            #print(str(sizeIn) + " " + str(sizeOut))
            self.layers.append(LogisticLayer(sizeIn, sizeOut, ind == 0, actives))
            logging.info("Layer %i, size [%i → %i], %s", ind, sizeIn, sizeOut, actives)
        #self.layers.append(LogisticLayer(layerConf[-1], 1, ind == 0, actives))
        #logging.info("Layer %i, size [%i → %i], %s", ind + 1, layerConf[-1], 1, actives)

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
        loss = DEE
        loss.errorString()
        # ----------------------------------
        if verbose:
            logging.info("LogRes using [%s] with %i epochs", loss.errorString, self.epochs)
        learned = False
        epoch = 1
        delta = 0
        REVERSED_CLASSIFIER_LAYER = 0 # classifier layer is the first in the reverse list
        trainingCost = []
        weightRatio = []
        while not learned:
            totalError = 0
            i = 0
            for input, label in zip(self.trainingSet.input, self.trainingSet.label):

                ##
                ## Feed forward step - let the layers evaluate input in a cascading manner
                ## The second parameter is used to apply dropout during training.
                ##
                output = self.forward(input, True)
                #totalError -= loss.calculateError(label, output)

                ##
                ## Backpropagation step - From back to front, calculate nextDerivates
                ## and pass the weights to the next layer
                ##
                layerStats = []
                error = loss.calculateError(np.array(label), self.layers[-1].output)
                #error = (np.array(label) - np.array(self.layers[-1].output))

                for ind, layer in enumerate(reversed(self.layers)):
                    #print(str(ind) + str(layer.shape))
                    if ind == REVERSED_CLASSIFIER_LAYER:
                        # The classifier layer. NOTE: Probably needs specific arguments
                        # depending on the error function!
                        layer.computeDerivative(error, None, None)
                        totalError -= error
                    else:
                        # hidden layer: propagate backwards with the derivates and the weights
                        # The minus sign indicates going from right to left index-wise.
                        #print(str(self.layers[-ind].weights.shape))
                        layer.computeDerivative(None, self.layers[-ind].delta, self.layers[-ind].weights)

                    layerStats.append("l: " + str(layer.shape) + " mean: " + '{0:.2f}'.format(np.average(layer.weights)) + " std: " + '{0:.2f}'.format(np.std(layer.weights)))
                if verbose and epoch % 10 == 1 and i < 3:
                    stats = ""
                    for statsLayer in reversed(layerStats):
                        stats += str(statsLayer) + " "
                    logging.info("%s", stats)

                ##
                ## Update step
                ##
                ratios = []
                for ind, layer in enumerate(self.layers):
                    if ind != 0:
                        layer.updateWeights(self.layers[ind - 1].output, self.learningRate)
                        #ratios.append(layer.weightUpdateRatio)
                ratios = np.array(ratios)
                # should be numbers of 1e-3
                # print(str(np.average(abs(ratios))))
                i += 1
            # stats about training cost
            trainingCost.append(sum(abs(totalError)))
            delta = trainingCost[-1] if epoch < 2 else abs(trainingCost[-1] - trainingCost[-2])

            acc = self.evaluate()
            acc = 1.0 - (1.0 * np.sum(acc) / len(acc))
            if verbose:
                logging.info("Epoch: %i; Acc: %f; Error: %f; ΔE: %f", epoch, acc, sum(abs(totalError)), delta)
            if epoch >= self.epochs or delta <= 1e-8:
                learned = True

            epoch = epoch + 1

        # 'python-tk package not found?' => apt-get install python-tk
        plt.plot(trainingCost, '-b',  label='Training Set')
        plt.ylabel('Error cost')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        # enable this to show graph
        plt.show()
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
        r = self.forward(temp)
        return r[0] > 0.5 and np.argmax(r) == 0
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


    def forward(self, input, training=False):
        output = input
        for layer in self.layers:
            if not training:
                output = layer.forward(output)
            else:
                # perform dropout if not classification
                output = layer.forward(output, layer.isClassifierLayer == False)



        return output
