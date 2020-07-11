# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary pay	ckages
import numpy as np
import argparse
import time
import cv2
import os
import scipy
from scipy import signal


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

shares=int(input("Enter no. of shares you want: "))

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
cv2.imwrite("input.png",image)
list_images=[]
for i in range(0,shares-1):
	list_images.append(cv2.imread(args["image"]))
(H, W) = image.shape[:2]



# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
#blobb=blob.reshape(blob.shape[2],blob.shape[3],blob.shape[1])
#blobb.astype("uint8")
#cv2.imshow("blob",blobb)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []



# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			list_shares_red=[]
			list_shares_green=[]
			list_shares_blue=[]
			for i in range(0,shares-1):
				mat1=np.random.randint(256,size=(W,H))
				mat2=np.random.randint(256,size=(W,H))
				mat3=np.random.randint(256,size=(W,H))
				list_shares_red.append(mat1)
				list_shares_green.append(mat2)
				list_shares_blue.append(mat3)
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])
f1=open("input.txt","w")
# ensure at least one detection exists
if len(idxs) > 0:
	p=0
	cv2.imshow("mainimage",image)
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates

		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		
		w+=30
		h+=30
		x-=15
		y-=15

		
		if x<0:
			x=0
		if y<0:
			y=0
		if x+w>W:
			w=w-(x+w-W)
		if y+h>H:
			h=h-(y+h-H)

		str1= str(x)+" "+str(y)+" "+str(w)+" "+str(h)
		f1.write(str1+"\n")
		
		# draw a bounding box rectangle and label on the image
		encrypted_image=[]
		q=0
		for l in range(0,shares-1):
			for k in range(0,w):
				for j in range(0,h):
					image[y+j,x+k][0]=(int(image[y+j,x+k][0])-list_shares_red[l][x+k,y+j])%256
					list_images[l][y+j,x+k][0]=list_shares_red[l][x+k,y+j]
					image[y+j,x+k][1]=(int(image[y+j,x+k][1])-list_shares_green[l][x+k,y+j])%256
					list_images[l][y+j,x+k][1]=list_shares_green[l][x+k,y+j]
					image[y+j,x+k][2]=(int(image[y+j,x+k][2])-list_shares_blue[l][x+k,y+j])%256
					list_images[l][y+j,x+k][2]=list_shares_blue[l][x+k,y+j]	
# 			cv2.imshow(str(p)+"intermediateimage"+str(q),image)
			q+=1
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		'''for abc in range(0, shares-1):
			cv2.imshow(str(p)+"random"+str(abc), list_images[abc])
			cv2.imwrite(str(p)+"random"+str(abc)+".png", list_images[abc])
		p+=1'''
		
# show the output image

cv2.imshow("Image", image)
cv2.imwrite("o1.png", image)
for i in range(0,shares-1):
	cv2.imshow("image"+str(i), list_images[i])
	cv2.imwrite("o"+str(i+2)+".png",list_images[i])
# cv2.imwrite("rect.png",image)

cv2.waitKey(0)