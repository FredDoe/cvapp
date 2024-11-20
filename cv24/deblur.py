import os
import numpy as np
import torch
from PIL import Image
from skimage.restoration import wiener
from skimage import io, restoration
from torchvision.transforms.functional import to_tensor, to_pil_image
from cv24.utils import CV24ImageManipulator
from .mprnet import MPRNet


class Deblur24(CV24ImageManipulator):

    def load_model(self):
        """
        Load the pretrained MPRNet model once when the app starts
        and store it globally.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        deblur_model = MPRNet().to(device)

        chkpoint = os.path.join(
            os.path.dirname(__file__), "weights", "mprnet_deblurring.pth"
        )
        deblur_model.load_state_dict(torch.load(chkpoint, map_location=device))

        deblur_model.eval()

    def mprnet_deblur(self, imsrc):
        img = Image.open(imsrc).convert("RGB")
        img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)

        deblur_model = self.load_model()
        with torch.no_grad():
            restored = deblur_model(img_tensor)[0]

        restored = restored.clamp(0, 1).squeeze(0).cpu()  # Remove batch dimension
        restored_img = to_pil_image(restored)

        return self.enc_im_to_b64(restored_img)
