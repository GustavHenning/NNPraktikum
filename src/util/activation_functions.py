# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide


class Activation:
    """
    Containing various activation functions and their derivatives
    """
    # TODO
    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # 1 / (1 + e^x)
        return divide(1, (1 + exp(-netOutput)))

    @staticmethod
    def sigmoidPrime(netOutput):
        # source: https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions
        return Activation.sigmoid(netOutput) * (1 - Acitvation.sigmoid(netOutput))


    @staticmethod
    def tanh(netOutput):
        # source: https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions
        return (2 / (1 + exp(-2 * netOutput))) - 1

    @staticmethod
    def tanhPrime(netOutput):
        # source: https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions
        return 1 - Activation.tanh(netoutput)**2

    @staticmethod
    def rectified(netOutput):
        return lambda x: max(0.0, x)

    @staticmethod
    def rectifiedPrime(netOutput):
        # source: https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions
        return  0 if netOutput < 0 else 1

    @staticmethod
    def identity(netOutput):
        return lambda x: x

    @staticmethod
    def identityPrime(netOutput):
        # source: https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions
        return 1

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        netExp = [exp(n) for n in netOutput]
        sumExp = sum(netExp)
        return [(n / sum(netExp)) for n in netExp]

    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
