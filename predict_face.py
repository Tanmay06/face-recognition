"""This program loads the pre-trained model and uses it to recognise face."""

#importing required libraries
import cv2
from sklearn.externals import joblib
from sklearn.decomposition import PCA

#loading cascade file to detect face
face_case = cv2.CascadeClassifier('face_casecade.xml')
cam = cv2.VideoCapture(0)
#loading the model to recognise face
clf = joblib.load("nn_clf.joblib")

while True:
    #reading and, scaling images for accuracy
    ret, img = cam.read(0)
    Y, X, C = img.shape
    scaleY = 180
    scaleX = int(X * (scaleY / Y))
    img = cv2.resize(img,(scaleX,scaleY))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_case.detectMultiScale(img,1.1,5)#detecting face
    
    for(x,y,w,h) in face:
        test = cv2.resize(gray[y:y+h,x:x+w],(21,21))
        test = test.ravel().reshape(1,441)
        pred = clf.predict(test/255)#recognising predicted face
        if pred == 1: 
        	color = (255,0,0)
        else:
        	color = (0,0,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)            
    
    #showing the output as video feed 
    img = cv2.flip(img,1)
    cv2.imshow("face",img)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
