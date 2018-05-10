import features
import numpy as np
import scipy.optimize as opt
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


class NeuralNetworkClassifier():
    def __init__(self, inputNum, hiddenNum, outputNum, dataNum, l):
        self.input = inputNum  # without bias node
        self.hidden = hiddenNum  # without bias node
        self.output = outputNum
        self.dataNum = dataNum
        self.l = l

        self.inputActivation = np.ones((self.input + 1, dataNum))  # add bias node
        self.hiddenActivation = np.ones((self.hidden + 1, dataNum))  # add bias node
        self.outputActivation = np.ones((self.output, dataNum))

        self.bias = np.ones((1, dataNum))

        self.inputChange = np.zeros((self.hidden, self.input + 1))
        self.outputChange = np.zeros((self.output, self.hidden + 1))

        self.hiddenEpsilon = np.sqrt(6.0 / (self.input + self.hidden))
        self.outputEpsilon = np.sqrt(6.0 / (self.input + self.output))

        self.inputWeights = np.random.rand(self.hidden, self.input + 1) * 2 * self.hiddenEpsilon - self.hiddenEpsilon
        self.outputWeights = np.random.rand(self.output, self.hidden + 1) * 2 * self.outputEpsilon - self.outputEpsilon

    def setLambda(self, l):
        self.l = l

    def feedForward(self, thetaVec):
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = sigmoid(hiddenZ)

        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = sigmoid(outputZ)

        costMatrix = self.outputTruth * np.log(self.outputActivation) + (1 - self.outputTruth) * np.log(
            1 - self.outputActivation)
        regulations = (np.sum(self.outputWeights[:, :-1] ** 2) + np.sum(self.inputWeights[:, :-1] ** 2)) * self.l / 2
        return (-costMatrix.sum() + regulations) / self.dataNum

    def backPropagate(self, thetaVec):
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        outputError = self.outputActivation - self.outputTruth
        hiddenError = self.outputWeights[:, :-1].T.dot(outputError) * dsigmoid(self.hiddenActivation[:-1:])

        self.outputChange = outputError.dot(self.hiddenActivation.T) / self.dataNum
        self.inputChange = hiddenError.dot(self.inputActivation.T) / self.dataNum

        self.outputChange[:, :-1].__add__(self.l * self.outputWeights[:, :-1])
        self.inputChange[:, :-1].__add__(self.l * self.inputWeights[:, :-1])

        return np.append(self.inputChange.ravel(), self.outputChange.ravel())

    @timing
    def train(self, trainData, trainLabels, iteration=100):
        self.inputActivation[:-1, :] = trainData
        self.outputTruth = self.genTruthMatrix(trainLabels)
        thetaVec = np.append(self.inputWeights.ravel(), self.outputWeights.ravel())
        thetaVec = opt.fmin_cg(self.feedForward, thetaVec, fprime=self.backPropagate, maxiter=iteration)
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))


    def classify(self, feature):
        if feature.shape[1] != self.inputActivation.shape[1]:
            self.inputActivation = np.ones((self.input + 1, feature.shape[1]))
            self.hiddenActivation = np.ones((self.hidden + 1, feature.shape[1]))
            self.outputActivation = np.ones((self.output + 1, feature.shape[1]))
        self.inputActivation[:-1, :] = feature

        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = sigmoid(hiddenZ)

        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = sigmoid(outputZ)
        if self.output > 1:
            return np.argmax(self.outputActivation, axis=0).tolist()
        else:
            return (self.outputActivation>0.5).ravel()

    def genTruthMatrix(self, trainLabels):
        truth = np.zeros((self.output, self.dataNum))
        for i in range(self.dataNum):
            label = trainLabels[i]
            if self.output == 1:
                truth[:,i] = label
            else:
                truth[label, i] = 1
        return truth
