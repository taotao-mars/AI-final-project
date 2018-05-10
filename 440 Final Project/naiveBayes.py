from collections import Counter
import math
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

class NaiveBayesClassifier():
    def __init__(self, k,f):
        self.k = k
        self.f=f

    def calPriorDistribution(self, labels):
       
        self.dist = Counter(labels)
        self.prior = {}
        for l in labels:
            self.prior[l] = float(self.dist[l]) / len(labels)
        return self.prior

    def calConditionalProbabilities(self, data, labels):
  
        occurrence = {}
        
        for i in range(len(labels)):
            l = labels[i]
            feature = features.featuresExtract(data[i],self.f)
            if l not in occurrence:
                occurrence[l] = np.array(feature)
            else:
                occurrence[l] += np.array(feature)
        self.conds = {}

        for l in labels:
            self.conds[l] = np.divide(occurrence[l] + self.k, float(self.dist[l] + self.k * 2))
        return self.conds

    def calLogJointProbabilities(self, datum):
      
        logJoint = {}
        feature = features.featuresExtract(datum,self.f)
        for l in self.dist.keys():
       
            logConds = np.log(self.conds[l])
          
            logCondsC = np.log(1 - self.conds[l])
    
            logJoint[l] = np.sum(np.array(feature) * logConds, dtype=float)
            logJoint[l] += np.sum((1 - np.array(feature)) * logCondsC, dtype=float)

            logJoint[l] += math.log(self.prior[l])
        return logJoint

    @timing
    def train(self, trainData, trainLabels):
        self.calPriorDistribution(trainLabels)
        self.calConditionalProbabilities(trainData, trainLabels)

    def classify(self, testData):
        guess = []
        self.posteriors = []
        for datum in testData:
            post = self.calLogJointProbabilities(datum)
            guess.append(np.argmax(post.values()))
            self.posteriors.append(post)
        return guess
