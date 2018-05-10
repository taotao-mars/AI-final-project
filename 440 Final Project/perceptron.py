from collections import Counter
import features
import numpy as np
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

class PerceptronClassifier():
    def __init__(self, trainData, trainLabels,f):
        self.weights = {}
        weight = np.array([[0 for col in range(trainData[0].width)] for row in range(trainData[0].height)])
        for l in trainLabels:
            if l not in self.weights:
                self.weights[l] = weight
        self.labels = self.weights.keys()
        #feature extraction type
        #0->basic,1->advanced
        self.f=f

    @timing
    def train(self, trainData, trainLabels,iteration):
        for k in range(iteration):
            for i in range(len(trainLabels)):
                truth = trainLabels[i]
                feature = features.featuresExtract(trainData[i],self.f)
                predict = self.learnWeights(feature)
                if predict != truth:
                    wt = self.weights[truth] + feature
                    wp = self.weights[predict] - feature
                    self.weights[truth] = wt
                    self.weights[predict] = wp

    def learnWeights(self, feature):
        scores = {}
        for l in self.labels:
            scores[l] = np.sum(self.weights[l] * feature)
        return np.argmax(scores.values())

    def classify(self, testData):
        guess = []
        for datum in testData:
            feature = features.featuresExtract(datum,self.f)
            guess.append(self.learnWeights(feature))
        return guess
