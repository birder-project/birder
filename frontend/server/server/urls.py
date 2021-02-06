from django.urls import include
from django.urls import path

from . import views

urlpatterns = [
    # path("favicon.ico", serve, {"path": "favicon.ico", "document_root": "static/", "show_indexes": False}),
    path("health/", views.health),
    path("inference/", include("inference.urls")),
]
