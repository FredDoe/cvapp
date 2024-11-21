import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from PIL import Image


class CV24ImageManipulator:
    PREFIX = "data:image/png;base64,"

    def open_image(self, imsrc, gray: bool = False):
        """Open the image from path, InMemoryUploadedFile, or TemporaryUploadedFile."""
        if isinstance(imsrc, (InMemoryUploadedFile, TemporaryUploadedFile)):
            imsrc.seek(0)
            image_array = np.frombuffer(imsrc.read(), np.uint8)
            if gray:
                return cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif isinstance(imsrc, str):  # File path
            if gray:
                return cv2.imread(imsrc, cv2.IMREAD_GRAYSCALE)
            return cv2.imread(imsrc)
        else:
            raise ValueError("Unsupported file type")

    def enc_im_to_b64(self, imsrc):
        """Encode the image from various sources (PIL Image, InMemoryUploadedFile, TemporaryUploadedFile, or path) to a base64 string."""

        if isinstance(imsrc, (InMemoryUploadedFile, TemporaryUploadedFile)):
            # Reset the file pointer and read the content
            imsrc.seek(0)
            encoded_string = base64.b64encode(imsrc.read()).decode("utf-8")
            return self.PREFIX + encoded_string

        elif isinstance(imsrc, str):  # If a file path is provided
            with open(imsrc, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            return self.PREFIX + encoded_string

        elif isinstance(imsrc, Image.Image):  # If a PIL Image is provided
            buffer = BytesIO()
            imsrc.save(buffer, format="PNG")
            buffer.seek(0)
            encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
            return self.PREFIX + encoded_string

        elif isinstance(imsrc, np.ndarray):  # OpenCV image (numpy array)
            buffer = BytesIO()
            if len(imsrc.shape) == 2:  # Grayscale image
                pil_image = Image.fromarray(imsrc)
            elif len(imsrc.shape) == 3:  # Color image
                pil_image = Image.fromarray(imsrc[:, :, ::-1])  # BGR to RGB for PIL
            else:
                raise ValueError("Unsupported image shape")
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
            return self.PREFIX + encoded_string

        else:
            raise ValueError("Unsupported image source type")
