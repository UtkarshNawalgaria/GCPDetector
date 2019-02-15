"""
This program is the main program that is used to detect 
Ground control points in the given image to assist the drones
in stitching images together using these points as reference.

function main():
 input: 
  image <- this image is used for detection.
 output:
  crops <- binary images of potential targets found in the image.
  bbox <- this contains the coordinates for the top-left corner of the bounding box 
   the height and width of the box((50,50))

- Various helper functions have been implemented in the utils.py file.
- The dimension of the bounding box has been chosen by observing the output in various images (50,50).
- Most of the values have been chosen by observation and different methods can be improved to automate the task.

More advanced techniques can be used to further filter out the cropped images to get less potential targets.

As we get the targets, we can run them through a machine learning model, to get the probability of the target being
in the cropped image, and then we can store the bbox-coordinates.

@ UTKARSH NAWALGARIA
@ nawalgaria.utkarsh8@gmail.com
@ 9312144277
 
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import (
	show_image,
	draw_contours,
	check_concavity,
	draw_bbox,
	save_cropped_images,
	morphology
)

# function to perfor rgb thresholding.
def rgb_threshold(img,low_range, high_range):
	return cv2.inRange(img, low_range, high_range)


def area_filter(contours):
	""" 
	! The lower and upper bound have been hardcoded, but can be obtained automatically 
	! by calculating the area covered by the object in the image w.r.t the size of the image.

	Input: contours <- lsit of contours obtained from the image.
	Ouput: areas <- list of contours with area between the 
			given values.
	"""

	areas = []
	for cnt in contours:
		ar = cv2.contourArea(cnt)
		if ar>100 and ar<=500:
			areas.append(cnt)

	return areas


# function to get the centroids of the contours.
# the centroid of the L-shape will mostly be at the junction of L.
def get_centroid(contours):
	"""
	Input: contours <- list of contours obtained in the image.
	Output: centroids <- list of centroids for the contours obtained.

	Assumption: The centroid of the object is the junction of the L-shape, i.e., the GCP.
		    This centroid is used as the center point of the bounding box around the object.
	"""
	centroids = []
	for cnt in contours:
		m = cv2.moments(cnt)
		cx = int(m['m10']/m['m00'])
		cy = int(m['m01']/m['m00'])
		centroids.append((cx,cy))
	return centroids

# main function to detect gcp.
def main():

	# read the image
	# convert the image from BGR to RGB.
	img = cv2.cvtColor(cv2.imread("DJI_0086.JPG"), cv2.COLOR_BGR2RGB)
	blank_image = np.zeros(img.shape, np.uint8)

	# performing thresholding to get objects from the background.
	# we assume that the lower bound of white is 200.
	binary_map = rgb_threshold(img, np.array([200,200,200]), np.array([255,255,255]))
	# perform morphological operations, to close the holes in the thresholded image and remove the noise.
	morphed = morphology(binary_map,np.ones((5,5),np.uint8))

	# get contours in the image.
	# we are only taking the external contours.
	_, contours, _ = cv2.findContours(morphed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	# get contours within a particular area range.
	# filter the contours based on the area.
	contours = area_filter(contours)

	# check for concave contours.
	# remove all the convex contours.
	contours = check_concavity(contours)

	# get the centroid of the contours.
	# for our GCP, the centroid should be at the intersection
	# of the L-shape.
	centroids = get_centroid(contours)

	# It contains the coordinates of the top-left corner and width and height of the bounding box.
	# x <- c1-25, y <- c2-25
	# width <- w=50, height <- h=50
	bboxs = [(c1-25,c2-25,50,50) for c1,c2 in centroids]

	# crop out the potential images.
	_ , output = save_cropped_images(img, bboxs, 0)
	

if __name__ == '__main__':

	main()
