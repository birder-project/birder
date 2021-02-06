from django.urls import include
from django.urls import path
from rest_framework import routers

from . import views


class InferenceRootView(routers.APIRootView):
    """
    Root view of inference API
    """


class DocumentedRouter(routers.DefaultRouter):
    APIRootView = InferenceRootView


ROUTER = DocumentedRouter()
ROUTER.register("language", views.LanguageView, basename="language")
ROUTER.register("predict", views.PredictView, basename="predict")
ROUTER.register("synset", views.SynsetView, basename="synset")

urlpatterns = [path("v1/", include((ROUTER.urls, "inference"), namespace="v1"))]
