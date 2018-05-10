import samples
from perceptron import PerceptronClassifier

def testing(num):
    trainData = samples.loadImagesFile("data/digitdata/trainingimages", num, 28, 28)
    trainLabels = samples.loadLabelsFile("data/digitdata/traininglabels", num)
    testData = samples.loadImagesFile("data/digitdata/testimages", 1000, 28, 28)
    testLabels = samples.loadLabelsFile("data/digitdata/testlabels", 1000)
    validData = samples.loadImagesFile("data/digitdata/validationimages", 1000, 28, 28)
    validLabels = samples.loadLabelsFile("data/digitdata/validationlabels", 1000)

    perceptron=PerceptronClassifier(trainData, trainLabels,0)
    perceptron.train(trainData, trainLabels,10)
    print "***********************************"
    print "*************Test Data*************"
    guess=perceptron.classify(testData)
    samples.verify(perceptron, guess, testLabels)
    print "***********************************"
    print "************Valid Data*************"
    guess=perceptron.classify(validData)
    samples.verify(perceptron,guess,validLabels)


if __name__ == "__main__":
        sampleDigit=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
        sampleFace=[45,90,135,180,225,270,315,300,405,450]
        sample=sampleDigit
        for i in range(len(sample)):
            print str(10*(i+1))+"%% training data, %d" % sample[i]
            testing(sample[i])
            print "***********************************"
