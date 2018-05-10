import samples
import numpy as np
from neuralNetwork import NeuralNetworkClassifier

def testing(num):
    trainData = np.load("traindigitbasic.npy")
    trainLabels = samples.loadLabelsFile("data/digitdata/traininglabels", num)
    testData = np.load("testdigitbasic.npy")
    testLabels = samples.loadLabelsFile("data/digitdata/testlabels", 1000)
    validData = np.load("validationdigitbasic.npy")
    validLabels = samples.loadLabelsFile("data/digitdata/validationlabels", 1000)

    neural = NeuralNetworkClassifier(28 * 28, 50, 10, num, 3.5)
    neural.train(trainData[:,0:num], trainLabels, 100)
    print "*************Test Data*************"
    guess = neural.classify(testData)
    samples.verify(neural, guess, testLabels)
    print "***********************************"
    print "************Valid Data*************"
    guess = neural.classify(validData)
    samples.verify(neural, guess, validLabels)

if __name__ == "__main__":
        sampleDigit=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
        sampleFace=[45,90,135,180,225,270,315,300,405,450]
        sample=sampleDigit
        for i in range(len(sample)):
            print str(10*(i+1))+"%% training data, %d" % sample[i]
            testing(sample[i])
            print "***********************************"
