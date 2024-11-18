import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from PIL import Image


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


class ImageProcessor:
    PREFIX = "data:image/png;base64,"

    def open_image(self, img_source):
        """Open the image from path, InMemoryUploadedFile, or TemporaryUploadedFile."""
        if isinstance(img_source, (InMemoryUploadedFile, TemporaryUploadedFile)):
            img_source.seek(0)
            image_array = np.frombuffer(img_source.read(), np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif isinstance(img_source, str):  # File path
            return cv2.imread(img_source)
        else:
            raise ValueError("Unsupported file type")

    def enc_im_to_b64(self, image_source):
        """Encode the image from various sources (PIL Image, InMemoryUploadedFile, TemporaryUploadedFile, or path) to a base64 string."""

        if isinstance(image_source, (InMemoryUploadedFile, TemporaryUploadedFile)):
            # Reset the file pointer and read the content
            image_source.seek(0)
            encoded_string = base64.b64encode(image_source.read()).decode("utf-8")
            return self.PREFIX + encoded_string

        elif isinstance(image_source, str):  # If a file path is provided
            with open(image_source, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            return self.PREFIX + encoded_string

        elif isinstance(image_source, Image.Image):  # If a PIL Image is provided
            buffer = BytesIO()
            image_source.save(buffer, format="PNG")
            buffer.seek(0)
            encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
            return self.PREFIX + encoded_string

        else:
            raise ValueError("Unsupported image source type")
