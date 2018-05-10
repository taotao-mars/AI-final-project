import samples
import features
import numpy as np

if __name__ == "__main__":
    trainData = samples.loadImagesFile("data/digitdata/trainingimages", 5000, 28, 28)
    trainBF=features.batchExtract(trainData,0)
    np.save("traindigitbasic",trainBF)
    testData = samples.loadImagesFile("data/digitdata/testimages", 1000, 28, 28)
    testBF=features.batchExtract(testData,0)
    np.save("testdigitbasic",testBF)
    validData = samples.loadImagesFile("data/digitdata/validationimages", 1000, 28, 28)
    validBF=features.batchExtract(validData,0)
    np.save("validationdigitbasic",validBF)

    trainData = samples.loadImagesFile("data/facedata/facedatatrain", 451, 60, 70)
    trainBF=features.batchExtract(trainData,0)
    np.save("trainfacebasic",trainBF)
    testData = samples.loadImagesFile("data/facedata/facedatatest", 150, 60, 70)
    testBF=features.batchExtract(testData,0)
    np.save("testfacebasic",testBF)
    validData = samples.loadImagesFile("data/facedata/facedatavalidation", 301, 60, 70)
    validBF=features.batchExtract(validData,0)
    np.save("validationfacebasic",validBF)

    trainAF=features.batchExtract(trainData,1)
    np.save("trainfaceadvanced",trainAF)
    testAF=features.batchExtract(testData,1)
    np.save("testfaceadvanced",testAF)
    validAF=features.batchExtract(validData,1)
    np.save("validationfaceadvanced",validAF)

