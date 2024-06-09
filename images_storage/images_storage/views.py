import json
import requests
from django.http import HttpResponse
from django.shortcuts import render


def home(request):
    return render(request, 'home.html')


def hello(request):
    url = "http://127.0.0.1:8000/hello"
    response = requests.get(url)
    if response.status_code == 200:
        content = json.loads(response.content)["Hello"]
        return HttpResponse(f"Hello {content}")
    return HttpResponse("Not able to connect to model_api")


