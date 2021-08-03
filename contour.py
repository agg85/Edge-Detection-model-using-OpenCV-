import cv2 as cv
import numpy as np

################         on IMG         ################

img = cv.imread('flatlay.jpg')
img = cv.resize(img, (360,200), interpolation = cv.INTER_CUBIC)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges',canny)

# Laplacian (gradient-based)
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

# Sobel (gradient-based)
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)
# cv.imshow('Combined Sobel', combined_sobel)

threshold, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
# cv.imshow('Adaptive Thresh', adaptive_thresh)

contours, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(f'{len(contours)} contour(s) found:')
### RETR_LIST: all contours- heirarchical and external

cv.drawContours(img, contours, -1, (0,0,255), 1)
cv.imshow('Contours drawn', img)

cv.waitKey(0)

################         on VID         ################

# vid = cv.VideoCapture('video.mp4')
# while True:
# 	isTrue, frame = vid.read()
# 	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# 	adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
# 	contours, heirarchies = cv.findContours(adaptive_thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# 	cv.drawContours(frame, contours, -1, (0,0,255), 1)
# 	cv.imshow('Contours drawn', frame)
# 	if(cv.waitKey(20) & 0xFF==ord('d')):
# 		break
# vid.release()
# cv.destroyAllWindows
