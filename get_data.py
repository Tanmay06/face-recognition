import cv2
import os

face_case = cv2.CascadeClassifier('face_casecade.xml')
name = input("Enter name:")
cam = cv2.VideoCapture(0)
no = 1
while True:
    ret, img = cam.read()
    img = cv2.resize(img,(0,0),fx = 0.75,fy = 0.75)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_case.detectMultiScale(gray,1.1,5)
    
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        fname = "t_{0}{1}".format(no,".jpg")
        cv2.imwrite(os.path.join(name,fname),img[y:y+h,x:x+w])
        no +=1
                    
    img = cv2.flip(img,1)
    cv2.imshow("face",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
    
