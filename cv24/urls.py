from django.urls import path
from . import views

app_name = "cv24"

urlpatterns = [
    path("", views.detect_edges, name="detect-edges"),
    path("edges/", views.detect_edges, name="detect-edges"),
    path("ocr/", views.detect_edges, name="ocr"),
    path("deblur/", views.detect_edges, name="deblur"),
]
