import cv2
import imutils
import numpy as np
from colorpie.image_processing import ShapeDetector
im = cv2.imread('test.jpg')
# ret, thresh = cv2.threshold(im,127,255,0)
# im2, contours,hierarchy = cv2.findContours(thresh, 1, 2)
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# for c in range(417, 420):
#     x,y,w,h = cv2.boundingRect(contours[c])
#     cv2.imshow('cont ' + str(c), cv2.drawContours(im, contours, c, (0,255,0), 3))
#     cv2.imshow('rect ' + str(c), cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2))
# cv2.waitKey(0)

image = im
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", gray)
cv2.waitKey(0)
cv2.imshow("Image", blurred)
cv2.waitKey(0)
cv2.imshow("Image", thresh)
cv2.waitKey(0)

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

# loop over the contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    # M = cv2.moments(c)
    # cX = int((M["m10"] / M["m00"]) * ratio)
    # cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
