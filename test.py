#!/usr/local/bin/python2
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np
isSignificant = 0.8 #TN/FP Threshold

Result, P, classNames = aT.fileClassification("trainingData/test/drink_1.wav", "knnDE","knn")
winner = np.argmax(P) #pick the result with the highest probability value.
if P[winner] > isSignificant :
  print("File: drink_1.wav is in category: " + classNames[winner] + ", with probability: " + str(P[winner]))
else :
  print("Can't classify sound: " + str(P))