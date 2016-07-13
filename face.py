import cv2
import numpy as np
import sys

filedir = './static'

def getMark(name):
	imagePath = filedir + "/" + name
	cascPath = "./arg.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	scale = 0.9
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1/scale,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)
	print "Found {0} faces!".format(len(faces))
	name_list = []
	count = 0
	print faces
	print image.shape
	height = image.shape[0]
	width = image.shape[1]
	for (x, y, w, h) in faces:
		count += 1

		new_x = int(x + 0.5 * w - 0.975 * h * scale)
		new_y = int(y - 0.6 * h)
		new_w = int(1.95 * h * scale)
		new_h = int(1.95 * h)

		left = max(0,new_x)
		top = max(0,new_y)	
		right = min(new_x + new_w, width - 1)
		bottom = min(new_y + new_h, height - 1)

		new_x = left
		new_y = top
		new_w = right - left
		new_h = bottom - top

		rectf = (new_x,new_y,new_w,new_h)

		mask = np.zeros(image.shape[:2],np.uint8)

		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)      

		cv2.grabCut(image,mask,rectf,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)    

		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		img = image*mask2[:,:,np.newaxis]

		img2 = img[new_y:new_y+new_h,new_x:new_x+new_w]
		file_name = filedir + "/out_" + str(count) + "_" + name
		cv2.imwrite(file_name,img2)
		name_list.append(file_name)
	return name_list
	
