#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
import sys
import numpy as np


def main(layerConf="2,1", noStupid=False):
    # example: "2,3,1"
    if layerConf == None:
        layerConf="2,1"
    layerConf = map(int, layerConf.split(','))
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    #we cant use values for params and then have layer be dynamic
    layerConfig = layerConf
    logRes = LogisticRegression(data.trainingSet,
                                data.validationSet,
                                data.testSet,
                                layerConfig)
    print(str(noStupid))
    # Train the classifiers
    print("=========================")
    print("Training..")
    if not noStupid:
        print("\nStupid Classifier has been training..")
        myStupidClassifier.train()
        print("Done..")

    #print("\nPerceptron has been training..")
    #myPerceptronClassifier.train()
    #print("Done..")

    print("\nLogistic Regression has been training..")
    logRes.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    if not noStupid:
        stupidPred = myStupidClassifier.evaluate()
    logPred = logRes.evaluate()
    #perceptronPred = myPerceptronClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    if not noStupid:
        print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
        evaluator.printAccuracy(data.testSet, stupidPred)

    #print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    #evaluator.printAccuracy(data.testSet, perceptronPred)

    print("\nResult of the Logistic Regression recognizer:")
    evaluator.printAccuracy(data.testSet, logPred)


if __name__ == '__main__':
    noStupid = False
    layerConf=None
    if len(sys.argv) > 1:
        layerConf = sys.argv[1]
    for arg in sys.argv:
        if arg == "-ns":
            noStupid = True

    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)
    main(layerConf, noStupid)
