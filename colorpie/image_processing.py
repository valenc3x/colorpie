""" Image processing module """
import cv2
from sklearn.decomposition import RandomizedPCA


class ImageProcessing:
    """ Image processing methods """
    def __init__(self):
        pass

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
