import cv2
import numpy as np
import os
from sklearn import svm

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16
    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

target = np.concatenate((np.repeat("tanmay",149),np.repeat("kshitij",149)))
tan_data = []
net_data = []

print("calculating tanmay..")
os.chdir("tanmay")
names = os.listdir()
names.remove(".DS_Store")
tan_data = [hog(cv2.imread(x)) for x in names]

print("calculating kshitij..")
os.chdir("../kshitij")
names = os.listdir()
names.remove(".DS_Store")
net_data = [hog(cv2.imread(x)) for x in names]

os.chdir("..")

data = np.array((tan_data[:-1],net_data[:-1])).reshape(-1,64)
model = svm.SVC(C=10,decision_function_shape="ovo",gamma = 0.0001)
model.fit(data,target)


face_case = cv2.CascadeClassifier('face_casecade.xml')
"""cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    img = cv2.resize(img,(0,0),fx = 0.75,fy = 0.75)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_case.detectMultiScale(gray,1.1,5)
    cv2.putText(img,"Face Recognition",(0,0),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
    
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        hist = hog(img[y:y+h,x:x+w])
        res = model.predict([hist])

        if res[0] == "kshitij":
            cv2.putText(img,"kshitij",(x+h,y),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)

        elif res[0] == "tanmay":
            cv2.putText(img,"tanmay",(x+h,y),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
                            
    img = cv2.flip(img,1)
    cv2.imshow("face",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()"""

test = cv2.imread("test_2.jpg")
img = cv2.resize(test,(0,0),fx = 0.75,fy = 0.75)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face = face_case.detectMultiScale(gray,1.1,5)

for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        hist = hog(img[y:y+h,x:x+w])
        res = model.predict([hist])

        if res[0] == "kshitij":
            cv2.putText(img,res[0],(x+h,y),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)

        elif res[0] == "tanmay":
            cv2.putText(img,res[0],(x+h,y),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)

cv2.imshow("face",img)
