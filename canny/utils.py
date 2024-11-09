import os
import cv2
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from django.core.files.uploadedfile import InMemoryUploadedFile


class Base64EncodedImgGenerator:
    PREFIX = "data:image/png;base64,"

    def __init__(self) -> None:
        pass

    def enc_im_to_b64(self, data):
        barcode_image = self.generate(data)
        buffer = BytesIO()
        barcode_image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
        return self.PREFIX + encoded_string


def open_image(img_path: str):
    """Open the image"""
    return cv2.imread(img_path)


def apply_canny(img_path: str):
    """Apply Canny"""
    BASE_DIR = os.path.dirname(img_path)
    name, ext = os.path.splitext(os.path.basename(img_path))

    img = open_image(img_path)
    edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)
    plt.figure()
    plt.title(f"{name} - Canny")
    plt.imsave(
        os.path.join(BASE_DIR, f"{name}-canny{ext}"), edges, cmap="gray", format="png"
    )
    plt.imshow(edges, cmap="gray")
    plt.show()
