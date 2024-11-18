from django.urls import path
from . import views

app_name = "cv24"

urlpatterns = [
    path("", views.detect_edges, name="edges"),
    path("edges/", views.detect_edges, name="edges"),
    path("ocr/", views.perform_ocr, name="ocr"),
    path("deblur/", views.deblur_image, name="deblur"),
]
