import os
import cv2

faceDetection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDetectionTwo = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDetectionThree = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDetectionFour = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

directory = os.listdir("D:\\University\\Dissertation\\Database\\dave\\")
fileNo = 1


for folders in directory:
    folder = os.listdir("D:\\University\\Dissertation\\Database\\dave\\" + folders)
    os.chdir("D:\\University\\Dissertation\\Database\\dave\\" + folders)

    for pics in folder:
        frame = cv2.imread(pics)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = faceDetection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
        faceTwo = faceDetectionTwo.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
        faceThree = faceDetectionThree.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
        faceFour = faceDetectionFour.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)


        if len(face) == 1:
            faceFeatures = face

        elif len(faceTwo) == 1:
            faceFeatures = faceTwo

        elif len(faceThree) == 1:
            faceFeatures = faceThree

        elif len(faceFour) == 1:
            faceFeatures = faceFour

        else:
            faceFeatures = ""

        for (x,y,w,h) in faceFeatures:
            print("face found inf file: %s" %pics)
            gray = gray[y:y+h, x:x+w]

            try:

                out = cv2.resize(gray, (350,350))
                cv2.imwrite("D:\\University\\Dissertation\\Database\\completeDave\\%s\\%s.jpg" %(folders, fileNo), out)

            except:
                pass

        fileNo += 1
            
        
