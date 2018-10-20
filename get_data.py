import cv2
import os

face_case = cv2.CascadeClassifier('face_casecade.xml')
name = input("Enter name:")
cam = cv2.VideoCapture(0)
 
no = 1
while True:
    ret, img = cam.read(0)
    img = cv2.resize(img,(0,0),fx = 0.25,fy = 0.25)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_case.detectMultiScale(img,1.1,5)
    
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+85,y+85),(255,0,0),1)
        fname = "t_{0}{1}".format(no,".jpg")
        cv2.imwrite(os.path.join(name,fname),cv2.resize(gray[y:y+85,x:x+85],(0,0),fx = 0.25,fy = 0.25))
        no +=1
                    
    img = cv2.flip(img,1)
    cv2.imshow("face",img)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
    
