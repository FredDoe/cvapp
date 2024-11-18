import cv2
from PIL import Image
import pytesseract
from cv24.utils import ImageProcessor


class OCR(ImageProcessor):
    def __init__(self) -> None:
        pass

    def preprocess(self, image_path):
        """
        Preprocess the image for better OCR results.
        Converts the image to grayscale and applies binary thresholding.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Path to the preprocessed image.
        """
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(
            gray, 128, 255, cv2.THRESH_BINARY
        )  # Binary thresholding
        preprocessed_path = "preprocessed_image.png"
        cv2.imwrite(preprocessed_path, thresh)  # Save preprocessed image
        return preprocessed_path

    def apply_ocr(
        self,
        image_path,
        output_path="highlighted_image.png",
        preprocess=True,
        lang="eng",
    ):
        """
        Extract text from an image and mark recognized characters with bounding boxes.

        Args:
            image_path (str): Path to the image file.
            output_path (str): Path to save the image with highlighted text.
            preprocess (bool): Whether to preprocess the image before OCR.
            lang (str): Language for OCR. Default is 'eng' (English).

        Returns:
            str: Extracted text from the image.
        """
        try:
            # Optionally preprocess the image
            if preprocess:
                image_path = self.preprocess(image_path)

            # Read the image for highlighting
            img = cv2.imread(image_path)

            # Perform OCR to get text and bounding box data
            h, w, _ = img.shape
            data = pytesseract.image_to_boxes(Image.open(image_path), lang=lang)

            # Draw bounding boxes
            for box in data.splitlines():
                b = box.split()
                char, x1, y1, x2, y2 = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])
                # Tesseract coordinates are in a flipped y-axis; adjust for OpenCV
                y1, y2 = h - y1, h - y2
                cv2.rectangle(img, (x1, y2), (x2, y1), (0, 255, 0), 2)  # Green box
                cv2.putText(
                    img,
                    char,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # Save the highlighted image
            cv2.imwrite(output_path, img)
            return f"Text highlighted and saved to {output_path}"
        except Exception as e:
            return f"An error occurred: {e}"
