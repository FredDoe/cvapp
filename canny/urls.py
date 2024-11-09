from django.urls import path
from . import views

app_name = "canny"

urlpatterns = [
    path("", views.detect_edges, name="detect-edges"),
    path("edge-detection/", views.detect_edges, name="detect-edges"),
]
