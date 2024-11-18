from django.shortcuts import render
from django.http import HttpRequest
from canny.edges import EdgeDetector
from canny.ocr import OCR


def detect_edges(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        detector = EdgeDetector()
        output = detector.apply_canny(image)
        original = detector.enc_im_to_b64(image)
        context = {"original": original, "output": output}
        return render(request, "canny/edge.html", context)

    context = {}
    return render(request, "canny/edge.html", context)


def perform_ocr(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        ocr = OCR()
        output = ocr.apply_canny(image)
        original = ocr.enc_im_to_b64(image)
        context = {"original": original, "output": output}
        return render(request, "canny/ocr.html", context)

    context = {}
    return render(request, "canny/ocr.html", context)
