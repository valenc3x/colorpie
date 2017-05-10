""" Image processing module """
import cv2
import numpy as np
import imutils

from sklearn.decomposition import RandomizedPCA


class ImageProcessing:
    """ Image processing methods """
    def __init__(self):
        pass

    @staticmethod
    def image_ratio(image, M, ratio):
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        return cX, cY, image[40:165, 25:195]

    @staticmethod
    def resize_to_width(image, width):
        """ Resize image based on set width and mantain aspect ratio """
        ratio = float(width / image.shape[1])
        dim = (width, int(image.shape[0] * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize_image(image):
        image = (image / 255).flatten()
        return image
        # pca = RandomizedPCA(n_components=2040)

    @staticmethod
    def detect_shape(contour):
        """ Detects the shape of an image section based on contours and
                bounding rectangles of open CV
        """
        # Initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 3:
            shape = "triangle"
        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        return shape

    @staticmethod
    def crop_image(image):
        resized = imutils.resize(image, width=500)
        ratio = image.shape[0] / float(resized.shape[0])

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            M = cv2.moments(c)
            # if moment 00 of image is 0
            if M['m00'] == 0:
                continue
            shape = ImageProcessing.detect_shape(c)
            if shape not in ['rectangle', 'square']:
                continue
            cX, cY, cropped = ImageProcessing.image_ratio(image, M, ratio)
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
        #     cv2.putText(
        #         image, shape, (cX, cY),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5, (255, 255, 255), 2)
        #     cv2.imshow("image", image)
        # cv2.imshow('cropped', cropped)
        # cv2.waitKey(0)
        return cropped
