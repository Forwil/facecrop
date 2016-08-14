import cv2
import numpy as np
import sys
import time

filedir = './static'

def show(mask):
	ret = np.zeros(mask.shape[:2],np.uint8)
	for i in range(0,mask.shape[0]):
		for j in range(0,mask.shape[1]):
			if mask[i][j] == cv2.GC_PR_FGD:
				ret[i][j] = 200
			if mask[i][j] == cv2.GC_FGD:
				ret[i][j] = 255
			if mask[i][j] == cv2.GC_PR_BGD:
				ret[i][j] = 100
			if mask[i][j] == cv2.GC_BGD:
				ret[i][j] = 0
	return ret

def compare(a,b):
	return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

def getMax(faces):
	maxv = 0
	ret = None
	for (x,y,w,h) in faces:
		if w * h > maxv:
			ret = (x,y,w,h)
			maxv = w * h
	return ret

def drawMark(init,image,rectf):
	x,y,w,h = rectf	
	bh = image.shape[0]
	bw = image.shape[1]
	ret = init

	for i in range(0,bh):
		for j in range(0,bw):
			if compare(image[i][j],[0,0,0]):
				ret[i][j] = cv2.GC_BGD
			if compare(image[i][j],[0,0,255]):
				ret[i][j] = cv2.GC_FGD
			if compare(image[i][j],[255,0,0]):
				ret[i][j] = cv2.GC_PR_FGD
	return  ret

def drawMark2(image,rectf):
	x,y,w,h = rectf	
	pad = image.shape[1] / 50
	scale = 1.0 / 4
	x = x + int(w * scale)
	y = y + int(h * scale)
	w = int(w * (1 - 2 * scale))
	h = int(h * (1 - 2 * scale))
	bh = image.shape[0]
	bw = image.shape[1]
	ret = image

	ret[y:y+h,x:x+w] = cv2.GC_FGD
	ret[y+h:bh,x+w/2-pad*2:x+w/2+pad*2] = cv2.GC_FGD
	ret[bh-4*pad:bh,x/2:(x+w+bw)/2] = cv2.GC_FGD
	return  ret



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

	if image.shape[1] > 1000:
		scalar = 1000.0 / image.shape[1]	
		h = int(image.shape[1] * scalar)
		w = int(image.shape[0] * scalar)
		image = cv2.resize(image,(h,w))

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
	print faces
	print image.shape
	height = image.shape[0]
	width = image.shape[1]
	if len(faces) == 0:
		return []

	(x,y,w,h) = getMax(faces)

	new_x = int(x + 0.5 * w - 0.975 * h * scale)
	new_y = int(y - 0.6 * h)
	new_w = int(1.95 * h * scale)
	new_h = int(1.95 * h)

	left = max(0,new_x)
	top = max(0,new_y)	
	right = min(left + new_w, width - 1)
	bottom = min(top + new_h, height - 1)
	
	new_x = left
	new_y = top
	new_w = right - left
	new_h = bottom - top

	if new_w > new_h * scale:
		new_w = new_h * scale
	if new_h > new_w / scale:
		new_h = new_w / scale
	
	right = new_x + new_w
	bottom = new_y + new_h	

	image = image[top:bottom,left:right]
	image = cv2.resize(image,(hei,wid))
	scalar = wid * 1.0 / new_h
	x -= left
	y -= top
	x *= scalar
	y *= scalar
	w *= scalar
	h *= scalar
	print (x,y,w,h)

	rectf = (0,0,image.shape[1]-1,image.shape[0]-1)
	mask = np.zeros(image.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)      

	cv2.grabCut(image,mask,rectf,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)    

	print "init mask" + str(time.time())	
	mask_img_read = cv2.imread("./mask.bmp")
	print "read mask done" + str(time.time())	
	mask_img = cv2.resize(mask_img_read,(image.shape[1],image.shape[0]))
	print "resize mask done" + str(time.time())	
	mask = drawMark(mask,mask_img,(x,y,w,h))
	print "init mask done" + str(time.time())	
	out = show(mask)
	cv2.imwrite("mask.jpg",out)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)      

	print "start cut" + str(time.time())	
#	rectf = (int(x),int(y),int(w),int(h))
#	mask = drawMark2(mask,(x,y,w,h))
	cv2.grabCut(image,mask,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)    
	print "end cut" + str(time.time())	
#	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
#	mask = cv2.erode(mask,kernel)
#	mask = cv2.dilate(mask,kernel)
#	mask = cv2.medianBlur(mask,9)

	bg = np.zeros(image.shape,np.uint8)
	mask2 = np.where(((mask == 2)|(mask==0)),0,1).astype('uint8')
	mask3 = np.where(((mask == 2)|(mask==0)),1,0).astype('uint8')
	bg[:,:] = color
	bg = bg * mask3[:,:,np.newaxis]
	img = image*mask2[:,:,np.newaxis]
	img = img + bg

#	img2 = img[new_y:new_y+new_h,new_x:new_x+new_w]
#	img3 = cv2.resize(img2,(hei,wid))
#		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
#		img3 = cv2.dilate(img3,kernel)
#	img3 = cv2.medianBlur(img3,3)
	img4 = join(img,rotate)
	file_name = filedir + "/out_" + str(time.time())+"_" + name
	cv2.imwrite(file_name,img4)
	name_list.append(file_name)
	file_name = filedir + "/out_" + str(time.time())+ "_single_" + name
	cv2.imwrite(file_name,img)
	name_list.append(file_name)
	print "end all" + str(time.time())	

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
