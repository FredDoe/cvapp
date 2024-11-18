import cv2
from PIL import Image
from PIL import Image
from canny.utils import ImageProcessor


class EdgeDetector(ImageProcessor):
    def apply_canny(self, img_source):
        """Apply Canny edge detection and return the image as a base64 encoded string."""
        img = self.open_image(img_source)
        edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)

        # Convert the edge-detected image to RGB for base64 encoding
        # edges= cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edge_image = Image.fromarray(edges)

        return self.enc_im_to_b64(edge_image)
