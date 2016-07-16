import cv2
import numpy as np
import sys

filedir = './static'

def getMark(name,wid = 413,hei = 295 ,color_id = 2,rotate = 0):
	if color_id == 0:
		color = (255,255,255)
	if color_id == 1:
		color = (0,0,255)
	if color_id == 2:
		color = (219,142,67)
	imagePath = filedir + "/" + name
	cascPath = "./arg.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	scale = hei * 1.0 / wid
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
	if len(faces) == 0:
		return []
	for (x, y, w, h) in faces[:1]:
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

		bg = np.zeros(image.shape,np.uint8)
		mask2 = np.where((mask == 2)|(mask==0),0,1).astype('uint8')
		mask3 = np.where((mask != 2)|(mask==0),0,1).astype('uint8')
		bg[:,:] = color
		bg = bg * mask3[:,:,np.newaxis]
		img = image*mask2[:,:,np.newaxis]
		img = img + bg

		img2 = img[new_y:new_y+new_h,new_x:new_x+new_w]
		img3 = cv2.resize(img2,(hei,wid))
		img4 = join(img3,rotate)
		file_name = filedir + "/out_" + str(count) + "_" + name
		cv2.imwrite(file_name,img4)
		name_list.append(file_name)
		file_name = filedir + "/out_" + str(count) + "_single_" + name
		cv2.imwrite(file_name,img3)
		name_list.append(file_name)

	return name_list

def join(img,rotate):
	width = 1795
	height = 1205
	if rotate == 1:
		tmp = width
		width = height
		height = tmp
	sub_height = img.shape[0]
	sub_width = img.shape[1]
	print img.shape
	offset = 20
	row = width / (sub_width + offset)
	col = height / (sub_height + offset)
	start_x = (width - (sub_width + offset) * row - offset) / 2
	start_y = (height - (sub_height + offset) * col - offset) / 2
	bg = np.zeros([height,width,3],np.uint8)
	bg[:,:] = (255,255,255)
	for i in range(0, row):
		for j in range(0, col):
			x = start_x + (sub_width + offset) * i
			y = start_y + (sub_height + offset) * j
			bg[y:y+sub_height,x:x+sub_width] = img
	
	return bg
