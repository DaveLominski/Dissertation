import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.externals import joblib


emotions = ["happy", "neutral", "sad", "surprised"] # "afraid", "angry", "disgusted", 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\dlib-19.9\\shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='poly', probability=True, tol=1e-3)

data = {}
'''This function has been written by Paul Van Gent from www.paulvangent.com'''
def getFiles(emotion):
    
    files = glob.glob("D:\\University\\Dissertation\\Database\\completeJoint\\%s\\*" %emotion)
    random.shuffle(files)
    training = files
  

    return training
'''This function has been written by Paul Van Gent from www.paulvangent.com'''
def getLandmarks(image):
    detections = detector(image,1)
    for k,d in enumerate(detections):

        shape = predictor(image,d)
        xlist = []
        ylist = []

        for i in range(1,68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []

        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y,x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"

'''This function has been written by Paul Van Gent from www.paulvangent.com'''
def makeSets():

    training_data = []
    training_labels = []

    for emotion in emotions:

        print("working on %s" %emotion)
        training= getFiles(emotion)


        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            getLandmarks(clahe_image)
            if data["landmarks_vectorised"] == "error":
                print("No face detected")
            else:
                training_data.append(data['landmarks_vectorised'])
                training_labels.append(emotions.index(emotion))

                    
    return training_data, training_labels



training_data, training_labels = makeSets()
nparTrain = np.array(training_data)
a = clf.fit(nparTrain, training_labels)
print(a)
joblib.dump(clf, 'trained4PolyData.pkl')
print("Dataset trained saved")



"""
clf.predict(prediction_data)
prob = clf.predict_proba(prediction_data)
prob_s = np.around(prob, decimals=3)
prob_s = prob_s*100
pred = clf.predict(prediction_data)

for count, i in enumerate(pred):
    print("Prediction: ", emotions[i])
    print ('Probability: ')
    print ('Surprised: ', prob_s[count,3])
    print ('Neutral: ', prob_s[count,1])
    print ('Happy: ', prob_s[count,0])
    print ('Sad: ', prob_s[count ,2])"""

