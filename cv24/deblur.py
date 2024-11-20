import cv2
import numpy as np
from skimage.restoration import wiener
from skimage import io, restoration
from cv24.utils import CV24ImageManipulator


class Deblur24(CV24ImageManipulator):
    def wiener_kernel(self, length, angle):
        kernel = np.zeros((length, length))
        center = length // 2
        angle = angle % 180
        radian = np.deg2rad(angle)

        for i in range(length):
            x = int(center + (i - center) * np.cos(radian))
            y = int(center + (i - center) * np.sin(radian))
            kernel[x, y] = 1

        return kernel / kernel.sum()

    def wiener_deblur(self, imsrc, kernel_len=15, kernel_angle=45, noise_power=0.01):
        # Load the blurred image
        image = self.open_image(imsrc, gray=True)
        # image = cv2.imread(imsrc, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError("Image not found at the specified path.")

        # Generate the motion blur kernel
        psf = self.wiener_kernel(kernel_len, kernel_angle)

        # Perform Wiener deconvolution
        deblurred = wiener(image, psf, balance=noise_power)

        # Normalize and convert to 8-bit image
        deblurred = np.clip(deblurred * 255, 0, 255).astype("uint8")

        return self.enc_im_to_b64(deblurred)

    def rich_lucy_blind_deblur(self, imsrc, iterations=30):
        # Load the blurred image
        image = io.imread(imsrc, as_gray=True)

        # Estimate the Point Spread Function (PSF)
        psf = np.ones((5, 5)) / 25  # Initial guess

        # Apply Richardson-Lucy deconvolution
        deblurred = restoration.richardson_lucy(image, psf, num_iter=iterations)

        return self.enc_im_to_b64(deblurred)

    def deepblur(self):
        pass

    def deblurgan(self):
        pass
