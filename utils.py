""" 
This program contains the helper functions for the main program.

"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# this is the path of the crops folder where the cropped images would be stored.
PATH = 'crops'

def save_cropped_images(img, bboxs, cntr):
	"""
	This function is used to crop out the roi's and then save them to a folder.
	We save only those images which have no.of contours after thresholding <5

	Input: img <- image
	       bboxs <- list of all the bounding boxes detected.
	       cntr <- counter used to name the cropped images.

	"""
	output = [] # stores the bbox-coordinate of the cropped images.
	for bbox in bboxs:
		x,y,w,h = bbox
		crop = img[y:y+h,x:x+w].copy()
		blank_image = np.zeros((50,50), np.uint8)
		if(len(crop))>0:
			yes, thresh = check_cnts(crop)
			if yes:
				cv2.imwrite(os.path.join(PATH,'crop'+str(cntr)+'.png'), thresh)
				output.append(bbox)
				cntr+=1
				print(cntr,end=' ')

	return cntr, output



def check_cnts(img):
	"""
	This function returns the cropped images that have less than 5 contours.

	Assumption: The number 5 is also based on observations. 
	"""

	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	ret2,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	_, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	contour = 0
	if len(contours)<=5:
		# contour = max_cnt_area(contours)
		return (True,thresh)
	else:
		return (False,thresh)


# perform morphological operations to fill holes and remove noise.
def morphology(binary_map,kernel):
	close = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE,kernel)
	opening = cv2.morphologyEx(close, cv2.MORPH_OPEN,kernel)

	return opening

# function to show an image.
def show_image(image):
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.imshow('image',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# plt.imshow(image); plt.show()

# function to draw contours on a blank image.
def draw_contours(blank_image, contours):
	conts  = cv2.drawContours(blank_image, contours, -1, (255,255,255), 1)
	return conts

# function to get all the concave contours.
def check_concavity(contours):
	concave = []
	for cnt in contours:
		approx = cv2.approxPolyDP(
			cnt, 0.01*cv2.arcLength(cnt,True),True
		)

		if not cv2.isContourConvex(approx):
			concave.append(cnt)
	return concave

# function to build the bbox on the original image.
def draw_bbox(img,centroids):
	w,h=50,50
	for center in centroids:
		x,y = center[0]-25, center[1]-25
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 1)

