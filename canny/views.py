from django.shortcuts import render
from django.http import HttpRequest
from .utils import apply_canny, ImageProcessor


def detect_edges(request: HttpRequest):
    if request.method == "POST":
        image = request.FILES.get("image")
        processor = ImageProcessor()
        output = processor.apply_canny(image)
        original = processor.enc_im_to_b64(image)
        if image.__class__.__name__ == "InMemoryUploadedFile":
            pass
        elif image.__class__.__name__ == "TemporaryUploadedFile":
            pass
        context = {"original": original, "output": output}
        return render(request, "canny/detect_edge.html", context)

    context = {}
    return render(request, "canny/detect_edge.html", context)
