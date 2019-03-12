"""This program collects all images data from positive and negative labels, and converts it into a csv file."""

#importing required libraries
import cv2
import numpy as np
import os 


def get_gray_values(image_name):
	#method to open, read and ravel the image contents
	#image_name: name of the image file 

	img = cv2.imread(image_name,0)
	img = img.ravel()
	return img

def get_gray_array(names,pixels,label):
	#method to open, read and ravel an array of images.
	#names: array of names of images
	#pixels: resolution in which the image is to be reshaped, pixels = h * w, (h,w)=>(1,pixels)
	#label: label to be given to the image, positive/negative => 1/0

	#reading images
	gray_array = np.zeros(pixels)
	gray_array = gray_array.reshape(1,pixels)

	for name in names:
		img_val = get_gray_values(name)
		gray_array = np.append(gray_array,img_val.ravel().reshape(1,pixels),axis=0)

	#appending label to images data array
	gray_array = gray_array[1:,:]
	y_vec = np.zeros((len(names),1)) + label
	gray_array = np.append(gray_array,y_vec,axis=1)
	return gray_array

#loading negative images 
neg_dirs = os.listdir("negative")
if ".DS_Store" in neg_dirs:
		neg_dirs.remove(".DS_Store")		

os.chdir("negative")
for n_dir in neg_dirs:
	names = os.listdir(n_dir)
	os.chdir(n_dir)

	if ".DS_Store" in names:
		names.remove(".DS_Store")		
	if neg_dirs.index(n_dir) is 0:
		neg_data = get_gray_array(names, 441, 0)
	else:
		neg_data = np.append(neg_data, get_gray_array(names,441,0),axis=0)
	os.chdir("..")

print("Negative set ready.")
os.chdir("..")

#loading positive images
pos_dirs = os.listdir("positive")
if ".DS_Store" in pos_dirs:
		pos_dirs.remove(".DS_Store")		

os.chdir("positive")
for p_dir in pos_dirs:
	names = os.listdir(p_dir)
	os.chdir(p_dir)

	if ".DS_Store" in names:
		names.remove(".DS_Store")		
	if pos_dirs.index(p_dir) is 0:
		pos_data = get_gray_array(names, 441, 1)
	else:
		pos_data = np.append(pos_data, get_gray_array(names,441,1),axis=0)
	os.chdir("..")

print("Positive set ready.")
os.chdir("..")

#combining negative and positive datasets
data_set = np.append(pos_data,neg_data,axis=0)
print("Mearged data.")

#saving dataset to csv file
np.savetxt("data_set.csv",data_set,delimiter=",")
print("File ready.")
