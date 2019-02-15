### GCP_detector

I have used basic image processing techniques to get the object that is required to get the GCP.
For this task, I just store the cropped images of potential targets in a folder and then iterate over these images to 
predict the image with the target using the trained model and then store the bbox-coordinates of the final targets.  


### Preprocess:
# Train an image classifier using binary maps of the targets.
	
	Given the dataset of images and their gcp coordinates:
		1. Create a bounding box around the target with coordinates as the center of the image.
		2. Crop the image.
		3. Use OTSU_BINARIZATION to get a binary image.
		4. Train Neural Network to recognize the L-shape target.
		5. Save the model.

### Target Detection:
	
	1. Read image
	2. Convert BGR to RGB
	3. Threshold the RGB image with the threshold value of (200,200,200) (I assume that the lower bound of white is 200 for every channel.)
	4. We use the opening and closing operations to remove noise, close small holes in the binary map and smooth out the edges.
	5. Get the contours in the image.
	6. Filter the contours based on the area. lower_bound=100, upper_bound=500(based on observations.)
	7. Filter the contours based on their convexity. Get only those contours that are concave.
	8. Get the centroids of all the contours.
	9. The centroids are used to get the bounding box for the objects.
	10. For image in crops:
			1. Threshold image using OTSU_BINARIZATION.
			2. Get the contours.
			3. if len(contours)<=5:
				a) save the thresholded image.
				b) save the bbox-coordinates.


### Postprocess:
# Use the trained model to predict target probability.

 	From the 'CROP' folder:
 		1.For image in CROP:
 			prob = predict(image)
 			if prob>0.75:
 				save image to final folder.







