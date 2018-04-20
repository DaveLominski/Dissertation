import cv2
import random
import glob
import numpy as np

fish = cv2.face.FisherFaceRecognizer_create()
emotions = [ "happy", "neutral", "surprised", "afraid", "angry", "disgusted", "sad"]

def retrieveFiles(emotion):
    files = glob.glob("D:\\University\\Dissertation\\Database\\completeDataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    predicting = files[-int(len(files)*0.2):]
    return training, predicting

def makeSets():
    trainingData = []
    trainingLabels = []
    predictionData = []
    predictionLabels = []

    for emotion in emotions:
        training, predicting = retrieveFiles(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            trainingData.append(gray)
            trainingLabels.append(emotions.index(emotion))

        for item in predicting:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            predictionData.append(gray)
            predictionLabels.append(emotions.index(emotion))
        
        
    return trainingData, trainingLabels, predictionData, predictionLabels

def runRecognizer():
    trainingData, trainingLabels, predictionData, predictionLabels = makeSets()

    print("training fisher face classifier")
    print("The size of the training set is:", len(trainingLabels), "images")
    fish.train(trainingData, np.asarray(trainingLabels))

    print("Prediction classification set")
    count = 0
    correct = 0
    incorrect = 0
    for image in predictionData:
        pred, conf = fish.predict(image)
        if pred == predictionLabels[count]:
            correct += 1
            count += 1
        else:
            cv2.imwrite("D:\\University\\Dissertation\\Database\\WRONG\\%s_%s_%s.jpg" %(emotions[predictionLabels[count]], emotions[pred], count), image)
            incorrect += 1
            count += 1
    return ((100*correct)/(correct+incorrect))


metascore = []
for i in range(0,10):
    correct = runRecognizer()
    print("Got", correct, "% correct")
    metascore.append(correct)

print("\n\nEnd score:", np.mean(metascore), "% correct")

