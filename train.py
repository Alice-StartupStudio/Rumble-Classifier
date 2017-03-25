#!/usr/local/bin/python2
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np

# Use featureAndTrain(listOfDirs, mtWin, mtStep, stWin, stStep, classifierType, modelName, computeBEAT) from audioTrainTest.py
# listOfDirs: TrainingData dir, 
# mtWin, mtStep, stWin, stStep: mid-term/short-term window size and step
# classifierType: svm, knn, extratrees, gradientboosting, randomforest
# modelName: self-named modelName, 
# computeBEAT: True if the long-term beat-related features
# Result: print result in each steps and chart of accuracy, save the model to same folder


# # Different models

aT.featureAndTrain(["trainingData/delia","trainingData/eating","trainingData/drink"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnDE", False)
# aT.featureAndTrain(["trainingData/delia","trainingData/eating","trainingData/drink"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "etDE", False)
# aT.featureAndTrain(["trainingData/delia","trainingData/eating","trainingData/drink"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gbDE", False)
# aT.featureAndTrain(["trainingData/delia","trainingData/eating","trainingData/drink"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "rfDE", False)
# aT.featureAndTrain(["trainingData/delia","trainingData/eating","trainingData/drink"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmDE", False)

