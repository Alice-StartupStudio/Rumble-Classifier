#!/usr/local/bin/python2


from pyAudioAnalysis import audioTrainTest as aT
# print "hello1"
import numpy as np
# print "hello2"
import RPi.GPIO as GPIO
# print "hello3"
import time

isSignificant = 0.8 #TN/FP Threshold
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37,GPIO.OUT)


# for needs to be replaced with While
for i in range(1,10):
	# need to determine naming system for files + make sure they exist before running the classifer
	# + delete file on end of run
	Result, P, classNames = aT.fileClassification("trainingData/test/drink_1.wav", "knnDE","knn")
	winner = np.argmax(P) #pick the result with the highest probability value.
	if P[winner] > isSignificant :
	  print("File: drink_1.wav is in category: " + classNames[winner] + ", with probability: " + str(P[winner]))
	  GPIO.output(37, GPIO.HIGH)
	else :
	  print("Can't classify sound: " + str(P))
	time.sleep(5)
	GPIO.output(37,GPIO.LOW)
	time.sleep(5)
	  
  
