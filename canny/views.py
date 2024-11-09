from django.shortcuts import render
from django.http import HttpRequest
from .utils import apply_canny


def detect_edges(request: HttpRequest):
    image = request.POST.get("image")
    # processed = apply_canny("F:\cv_project\canny\static\canny\img\pixel9.png")
    context = {"orginal": image, "processed": "processed"}
    return render(request, "canny/detect_edge.html", context)
