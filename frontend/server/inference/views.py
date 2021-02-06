import requests
from django.conf import settings
from django.utils.translation import gettext as _
from rest_framework import parsers
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response

from . import serializers
from . import synset


class LanguageView(viewsets.ViewSet):
    """
    Return a list of supported languages
    """

    def list(self, _request):
        languages = settings.LANGUAGES
        languages = dict(languages)
        return Response(data=languages, status=status.HTTP_200_OK)


class PredictView(viewsets.ViewSet):
    """
    Returns translated top-5 predictions
    """

    parser_classes = (parsers.MultiPartParser,)

    def create(self, request):
        try:
            serializer = serializers.ImageSerializer(
                data={"image": request.FILES["image"], "model": request.data["model"]}
            )

        except KeyError:
            return Response(data={}, status=status.HTTP_400_BAD_REQUEST)

        serializer.is_valid(raise_exception=True)
        data = serializer.save()
        response = requests.post(f"{settings.MMS_PREDICT_URL}/{data['model']}", data["image"])
        prediction = response.json()

        for prob in prediction:
            prob["class"] = _(prob["class"])

        return Response(data=prediction, status=status.HTTP_200_OK)

    def list(self, _request):
        response = requests.get(f"{settings.MMS_MANAGEMENT_HOST}/models")
        raw_model_response = response.json()
        models = []
        for item in raw_model_response["models"]:
            models.append(item["modelName"])

        return Response(data=models, status=status.HTTP_200_OK)


class SynsetView(viewsets.ViewSet):
    """
    Return synset array
    """

    def list(self, _request):
        classes = synset.SYNSET
        return Response(data=classes, status=status.HTTP_200_OK)
