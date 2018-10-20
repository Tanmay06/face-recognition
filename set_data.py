import cv2
import numpy as np
import os 


def get_gray_values(image_name):
	img = cv2.imread(image_name,0)
	img = img.ravel()
	return img

def get_gray_array(names,pixels,label):
	gray_array = np.zeros(pixels)
	gray_array = gray_array.reshape(1,pixels)
	for name in names:
		img_val = get_gray_values(name)
		gray_array = np.append(gray_array,img_val.ravel().reshape(1,pixels),axis=0)

	gray_array = gray_array[1:,:]
	y_vec = np.zeros((len(names),1)) + label
	gray_array = np.append(gray_array,y_vec,axis=1)
	return gray_array

os.chdir("ajji2")
names = os.listdir()
#names.remove(".DS_Store")
neg_data = get_gray_array(names, 441, 0)
print("Negative set ready.")
os.chdir("../tan2")
names = os.listdir()

pos_data = get_gray_array(names, 441, 1)
print("Positive set ready.")

data_set = np.append(pos_data,neg_data,axis=0)
print("Mearged data.")

os.chdir("..")
np.savetxt("data_set.csv",data_set,delimiter=",")
print("File ready.")
