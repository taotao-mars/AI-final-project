import samples
import numpy as np
from neuralNetwork import NeuralNetworkClassifier

def testing(num):
    trainData = np.load("trainfacebasic.npy")
    trainLabels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", num)
    testData = np.load("testfacebasic.npy")
    testLabels = samples.loadLabelsFile("data/facedata/facedatatestlabels", 151)
    validData = np.load("validationfacebasic.npy")
    validLabels = samples.loadLabelsFile("data/facedata/facedatavalidationlabels", 301)
    loop=True
    while loop:
        neural = NeuralNetworkClassifier(60 * 70, 500, 1, num, 0.03)
        neural.train(trainData[:,0:num], trainLabels, 100)
        print "************Test Data**************"
        guess = neural.classify(testData)
        loop=samples.verify(neural, guess, testLabels)
        if loop:
            continue
        print "***********************************"
        print "***********Valid Data**************"
        guess = neural.classify(validData)
        samples.verify(neural, guess, validLabels)


if __name__ == "__main__":
        sampleDigit=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
        sampleFace=[45,90,135,180,225,270,315,300,405,451]
        sample=sampleFace
        for i in range(len(sample)):
            print str(10*(i+1))+"%% training data, %d" % sample[i]
            testing(sample[i])
            print "***********************************"
