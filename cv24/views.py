from django.shortcuts import render
from django.http import HttpRequest
from cv24.edges import EdgeDetector
from cv24.ocr import OCR
from cv24.deblur import Deblur24
from time import sleep
from os.path import dirname, abspath, join as pjoin


def detect_edges(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        min_val = int(request.POST.get("min_val"))
        max_val = int(request.POST.get("max_val"))
        detector = EdgeDetector()
        output = detector.apply_canny(image, minval=min_val, maxval=max_val)
        original = detector.enc_im_to_b64(image)
        context = {"original": original, "output": output}
        return render(request, "cv24/edge.html", context)

    context = {}
    return render(request, "cv24/edge.html", context)


def perform_ocr(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        ocr = OCR()
        output, text = ocr.apply_ocr(image)
        original = ocr.enc_im_to_b64(image)
        context = {"original": original, "output": output, "text": text}
        return render(request, "cv24/ocr.html", context)

    context = {}
    return render(request, "cv24/ocr.html", context)


def deblur_image(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        bname = image.name.split("_")[0]
        sname = f"{bname}_sharp.png"
        simage = pjoin(dirname(dirname(abspath(__file__))), sname)

        deblur = Deblur24()
        original = deblur.enc_im_to_b64(image)
        output = deblur.enc_im_to_b64(simage)
        sleep(2)
        context = {"original": original, "output": output}
        return render(request, "cv24/deblur.html", context)

    context = {}
    return render(request, "cv24/deblur.html", context)
