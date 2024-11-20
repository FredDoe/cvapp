import cv2
from PIL import Image, ImageFilter
import numpy as np
from cv24.utils import CV24ImageManipulator


class EdgeDetector(CV24ImageManipulator):
    def apply_canny(self, imsrc):
        """Apply Canny edge detection and return the image as a base64 encoded string."""
        img = self.open_image(imsrc)
        edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)
        edge_image = Image.fromarray(edges)
        return self.enc_im_to_b64(edge_image)

    def apply_sobel(self, imsrc):
        img = self.open_image(imsrc)
        img = img.astype(np.uint8)
        edges = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
        edge_image = Image.fromarray(edges.astype(np.uint8))
        return self.enc_im_to_b64(edge_image)
