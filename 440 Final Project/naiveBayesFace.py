import samples
from naiveBayes import NaiveBayesClassifier

def testing(num):
    trainData = samples.loadImagesFile("data/facedata/facedatatrain", num, 60, 70)
    trainLabels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", num)
    testData = samples.loadImagesFile("data/facedata/facedatatest", 150, 60, 70)
    testLabels = samples.loadLabelsFile("data/facedata/facedatatestlabels", 151)
    validData = samples.loadImagesFile("data/facedata/facedatavalidation", 301, 60, 70)
    validLabels = samples.loadLabelsFile("data/facedata/facedatavalidationlabels", 301)

    nb = NaiveBayesClassifier(1,0)
    nb.train(trainData, trainLabels)
    print "***********************************"
    print "************Test Data**************"
    guess = nb.classify(testData)
    samples.verify(nb,guess,testLabels)
    print "***********************************"
    print "***********Valid Data**************"
    guess=nb.classify(validData)
    samples.verify(nb,guess,validLabels)


if __name__ == "__main__":
        sampleDigit=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
        sampleFace=[45,90,135,180,225,270,315,300,405,451]
        sample=sampleFace
        for i in range(len(sample)):
            print str(10*(i+1))+"%% training data, %d" % sample[i]
            testing(sample[i])
            print "***********************************"
