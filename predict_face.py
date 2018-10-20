import cv2
from sklearn.externals import joblib

face_case = cv2.CascadeClassifier('face_casecade.xml')
cam = cv2.VideoCapture(0)
clf = joblib.load("nn_clf.joblib")


while True:
    ret, img = cam.read(0)
    img = cv2.resize(img,(0,0),fx = 0.25,fy = 0.25)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_case.detectMultiScale(img,1.1,5)
    
    for(x,y,w,h) in face:
        test = cv2.resize(gray[y:y+85,x:x+85],(0,0),fx = 0.25,fy = 0.25)
        test = test.ravel().reshape(1,441)
        pred = clf.predict(test)
        if pred == 1: 
        	color = (255,0,0)
        else:
        	color = (0,0,255)
        cv2.rectangle(img,(x,y),(x+85,y+85),color,1)            
    
    img = cv2.flip(img,1)
    cv2.imshow("face",img)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
