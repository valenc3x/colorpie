import cv2
import imutils
import numpy as np
from colorpie import ShapeDetector
from colorpie import ArtGatherer


def main():
    card = ArtGatherer.card_info()
    image = cv2.imread('akroma.jpg')
    # image = card.image
    image = image[35:173, 17:205]
    # ratio = image.shape[0] / float(resized.shape[0])
    ratio = 1

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    cnts = cv2.findContours(
        gray.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
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
        # show the output image
        cv2.imshow(shape, image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
