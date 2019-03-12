"""program to collect face data. The program uses the device camera to record images. It leverages HAAR cascade to detect human face in the image. 
This detected faces are stored in local directory and used later to train the model."""

#importing required libraries
import cv2
import os

#loading cascade file
face_case = cv2.CascadeClassifier('face_casecade.xml')

#asking for label, positive/negative, of the data and name for the directory in whic hthe data is to be stored
label = input("Enter label(Positive/Negative):")
name = input("Enter name:")

#if directory not available creating the directory
if name not in os.listdir(label):
    os.mkdir(os.path.join(label,name))

#initialising cam input
cam = cv2.VideoCapture(0)
no = 1

#recording until esc in pressed
while True:
    ret, img = cam.read(0)
    Y, X, C = img.shape
    scaleY = 180
    scaleX = int(X * (scaleY / Y))
    img = cv2.resize(img,(scaleX,scaleY))#rescaling the image from the camera fro efficiency
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converting to grayscale
    face = face_case.detectMultiScale(img,1.1,5)#detecting the face using cascade
    
    #storing face in directory
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        fname = "t_{0}{1}".format(no,".jpg")
        cv2.imwrite(os.path.join(label,name,fname),cv2.resize(gray[y:y+h,x:x+w],(21,21)))
        no +=1
                    
    img = cv2.flip(img,1)
    cv2.imshow("face",img)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
    
