import cv2
from PIL import Image
import pytesseract
from cv24.utils import CV24ImageManipulator


class OCR(CV24ImageManipulator):
    def __init__(self) -> None:
        pass

    def preprocess(self, image):
        """
        Preprocess the image for better OCR results.
        Converts the image to grayscale and applies binary thresholding.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(
            gray, 128, 255, cv2.THRESH_BINARY
        )  # Binary thresholding
        return thresh

    def apply_ocr(self, img_src, preprocess=False, lang="eng"):
        """
        Extract text from an image and mark recognized characters with bounding boxes.
        """
        try:
            # Read the image for highlighting
            img = self.open_image(img_src)

            # Optionally preprocess the image
            if preprocess:
                img = self.preprocess(img)

            # Perform OCR to get text and bounding box data
            h, w, _ = img.shape
            data = pytesseract.image_to_boxes(img, lang=lang)

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
            # cv2.imwrite(output_path, img)
            # return f"Text highlighted and saved to {output_path}"
            image = Image.fromarray(img)
            return self.enc_im_to_b64(image)
        except Exception as e:
            return f"An error occurred: {e}"
