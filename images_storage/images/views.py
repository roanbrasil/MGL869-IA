import json
import requests
from pathlib import Path
from django.shortcuts import render, redirect

from images.forms import DeleteFormSet, UploadForm
from images.models import Image, CATEGORIES, PROCESSES


def images_list(request):
    """Display all user images."""
    images = Image.objects.filter(user=request.user, process=PROCESSES.TEST)
    return render(request, "images/images_list.html", {"images": images})


def images_list_by_category(request, category):
    """Display the user images by category."""
    images = Image.objects.filter(user=request.user, category=category, process=PROCESSES.TEST)
    return render(
        request,
        "images/images_list_by_category.html",
        {"category": CATEGORIES.names[category], "images": images},
    )


def classify_images(images: list[Image]) -> None:
    """Query the model API with the images for classification."""
    multiple_files = []
    for image in images:
        image_path = Path(image.src.path)
        multiple_files.append(('images', (image_path.name, image_path.read_bytes(), 'image/jpq')))
    url = "http://127.0.0.1:8000/classify_images"
    response = requests.post(url, files=multiple_files)
    if response.status_code == 200:
        for i, category in enumerate(json.loads(response.content)["categories"]):
            images[i].category = CATEGORIES(category)
            images[i].save()


def upload_images(request):
    """Upload the images to the database. Classify them by querying the model API"""
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            form.save()
            # Classify the new uploaded images
            uploaded_images = Image.objects.filter(
                user=request.user, process=PROCESSES.TEST, category=None
            )
            classify_images(uploaded_images)
            return redirect("images:list")
    return render(request, "images/upload_images.html", {"form": UploadForm})


def delete_images(request):
    """Delete the selected images from the database."""
    images = Image.objects.filter(user=request.user, process=PROCESSES.TEST)
    if request.method == "POST":
        formset = DeleteFormSet(request.POST, queryset=images)
        if formset.is_valid():
            formset.save()
            images = Image.objects.filter(user=request.user)
    return render(
        request,
        "images/delete_images.html",
        {"formset": DeleteFormSet(queryset=images)},
    )
