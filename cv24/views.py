from django.shortcuts import render
from django.http import HttpRequest
from cv24.edges import EdgeDetector
from cv24.ocr import OCR
from cv24.deblur import Deblur24
import os
from PIL import Image
from io import BytesIO
import torch
from skimage import img_as_ubyte
from torchvision.transforms.functional import to_tensor, to_pil_image

from django.http import JsonResponse


def detect_edges(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        detector = EdgeDetector()
        output = detector.apply_canny(image)
        original = detector.enc_im_to_b64(image)
        context = {"original": original, "output": output}
        return render(request, "cv24/edge.html", context)

    context = {}
    return render(request, "cv24/edge.html", context)


def perform_ocr(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        ocr = OCR()
        output = ocr.apply_ocr(image)
        original = ocr.enc_im_to_b64(image)
        context = {"original": original, "output": output}
        return render(request, "cv24/ocr.html", context)

    context = {}
    return render(request, "cv24/ocr.html", context)


def deblur_image(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        deblur = Deblur24()
        original = deblur.enc_im_to_b64(image)
        output = deblur.mprnet_deblur(image)
        context = {"original": original, "output": output}
        return render(request, "cv24/deblur.html", context)

    context = {}
    return render(request, "cv24/deblur.html", context)
