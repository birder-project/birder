from django.http import HttpResponse


def health(_request):
    return HttpResponse("healthy", content_type="text/plain")
