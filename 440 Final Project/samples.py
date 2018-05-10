from naiveBayes import NaiveBayesClassifier
import numpy as np
from perceptron import PerceptronClassifier
from neuralNetwork import NeuralNetworkClassifier

class Datum:
    def __init__(self, data, width, height):
        self.width = width
        self.height = height
        if data == None:
            data = [[' ' for i in range(width)] for j in range(height)]
        self.pixels = mapToInteger(data)

    def getPixel(self, row, col):
        return self.pixels[row][col]

    def getPixels(self):
        return self.pixels

def loadImagesFile(filename, n, width, height):
    images = []
    with open(filename, 'r') as f:
        for i in range(n):
            pixels = []
            for j in range(height):
                pixels.append(list(f.readline())[:-1])
            if len(pixels[0]) < width - 1:
                break
            images.append(Datum(pixels, width, height))
    return images


def loadLabelsFile(filename, n):
    labels = []
    with open(filename, 'r') as f:
        for i in range(n):
            line = f.readline()
            if (line == ""):
                break
            labels.append(int(line.strip()))
    return labels


def mapToInteger(data):
    if (type(data) != type([])):
        return convertPixel(data)
    else:
        return map(mapToInteger, data)


def convertPixel(char):
    if (char == ' '):
        return 0
    elif (char == '+'):
        return 1
    elif (char == '#'):
        return 2


def verify(classifier, guesses, testLabels):
    hit = 0
    for i in range(len(guesses)):
        predict = guesses[i]
        truth = testLabels[i]
        if predict == truth:
            hit += 1
    accuracy = float(hit) / len(guesses)
    if accuracy>0.6:
        print "***********************************"
        print "Total: %d examples" % len(guesses)
        print "Prediction: %d" % hit
        print "Accuracy: %f" % accuracy
        print ""
        return False
    else:
        return True
