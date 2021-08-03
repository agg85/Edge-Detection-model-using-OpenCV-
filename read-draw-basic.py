import cv2 as cv
import numpy as np

################         READ         ################

def rescaleFrame(frame, scale = 0.75):
	width = int(frame.shape[1] * scale)
	height = int(frame.shape[0] * scale)
	dimensions = (width,height)
	return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

img = cv.imread('flatlay.jpg')
cv.imshow('flatlay', rescaleFrame(img))
cv.waitKey(0)

capture = cv.VideoCapture('video.mp4')
while True:
	isTrue, frame = capture.read()
	cv.imshow('Video',frame)
	if(cv.waitKey(20) & 0xFF==ord('d')):
		break
capture.release()
cv.destroyAllWindows()

################         DRAW         ################

blank = np.zeros((500,500,3), dtype='uint8')
# cv.imshow('Blank',blank)

# 1. Paint the image a certain color
blank[:] = 0,255,255
blank[200:300, 200:300] = 255,0,0
# 2. Rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (0,255,0), thickness=-1)
# 3. Circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=3)
# 4. Line
cv.line(blank, (100,250), (300,400), (255,255,255), thickness = 3)
# 5. Write text on img
cv.putText(blank, 'Hello, my name is Aryan', (20,450), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,0), 2)

cv.waitKey(0)

################         BASIC         ################

img = cv.imread('flatlay.jpg')
cv.imshow('flatlay',img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
canny = cv.Canny(img, 125, 175)
dilated = cv.dilate(canny, (7,7), iterations=3)
eroded = cv.erode(dilated, (7,7), iterations=3)
resized = cv.resize(img, (500,500), interpolation = cv.INTER_CUBIC)
cropped = img[50:200, 200:400]

cv.waitKey(0)
