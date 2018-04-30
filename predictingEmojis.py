import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
import emoji
import sys
import os

clf = joblib.load("trainedPolyData.pkl")
emotions = ["afraid", "angry", "disgusted","happy", "neutral", "sad", "surprised"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\dlib-19.9\\shape_predictor_68_face_landmarks.dat")
data = {}

'''This function has been written by Paul Van Gent from www.paulvangent.com'''
def getFiles(emotion):
    
    files = glob.glob("D:\\University\\Dissertation\\Database\\completeDave\\happy\\19.jpg") # %emotion
    random.shuffle(files)
    prediction = files

    return prediction

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

    prediction_data = []
    prediction_labels = []
    prediction = getFiles('happy')

    for item in prediction:

        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        getLandmarks(clahe_image)
        if data['landmarks_vectorised'] == "error":
            print("No face detected")
        else:
            prediction_data.append(data['landmarks_vectorised'])
            prediction_labels.append(emotions.index('happy'))

    return prediction_data, prediction_labels

def prediction():
    accuracy = [] 

    prediction_data, prediction_labels = makeSets()
    nparPred = np.array(prediction_data)
    pred_lin = clf.score(nparPred, prediction_labels)
    print("Polynomial accuracy for test: " ,pred_lin)

    clf.predict(prediction_data)
    prob = clf.predict_proba(prediction_data)
    prob_s = np.around(prob, decimals=3)
    prob_s = prob_s*100
    pred = clf.predict(prediction_data)
        

    for count, i in enumerate(pred):
        print("Prediction: ", emotions[pred[count]])
        print ('Probability: ')
        print ('Surprised: ', prob_s[count,6])
        print ('Neutral: ', prob_s[count,4])
        print ('Happy: ', prob_s[count,3])
        print ('Sad: ', prob_s[count ,5])
        print ('Afraid: ', prob_s[count ,0])
        print ('Angry: ', prob_s[count ,1])
        print ('Disgusted: ', prob_s[count ,2])
            
        picPred = emotions[pred[count]]
    return picPred


def emoji():
    import emoji 
    pred = prediction()
    happy = emoji.emojize(':smile: :smiley: :laughing: :grin: :yum:' , use_aliases=True)
    sad = emoji.emojize(':sob: :cry: :disappointed: :weary: :worried:' , use_aliases=True)
    surprised = emoji.emojize(':hushed: :open_mouth: :disappointed_relieved: :scream_cat: :scream:' , use_aliases=True)
    afraid = emoji.emojize(':rage: :cold_sweat: :confounded: :open_mouth: :unamused:' , use_aliases=True)
    neutral = emoji.emojize(':no_mouth: :neutral_face: :expressionless: :sleeping: :eyes:' , use_aliases=True)
    angry = emoji.emojize(':rage: :triumph: :disappointed: :unamused: :pouting_cat:' , use_aliases=True)
    disgusted = emoji.emojize(':poop: :mask: :grimacing: :confounded: :persevere:' , use_aliases=True)
    print(pred)

    if pred == 'happy':
        with open('test.htm','w',encoding='utf-8-sig') as f:
            f.write(happy)
            os.startfile('test.htm')
    if pred == 'sad':
        with open('test.htm','w',encoding='utf-8-sig') as f:
            f.write(sad)
            os.startfile('test.htm')
    if pred == 'surprised':
        with open('test.htm','w',encoding='utf-8-sig') as f:
            f.write(surprised)
            os.startfile('test.htm')
    if pred == 'afraid':
            with open('test.htm','w',encoding='utf-8-sig') as f:
                f.write(afraid)
                os.startfile('test.htm')
    if pred == 'neutral':
            with open('test.htm','w',encoding='utf-8-sig') as f:
                f.write(neutral)
                os.startfile('test.htm')
    if pred == 'angry':
        with open('test.htm','w',encoding='utf-8-sig') as f:
            f.write(angry)
            os.startfile('test.htm')
    if pred == 'disgusted':
        with open('test.htm','w',encoding='utf-8-sig') as f:
            f.write(disgusted)
            os.startfile('test.htm')  

emoji()


            

