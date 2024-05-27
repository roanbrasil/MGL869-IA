from django.shortcuts import render, redirect

from images.forms import UploadForm
from images.models import Image, CATEGORIES

from django.conf import settings


def images_list(request):
    images = Image.objects.filter(user=request.user)
    return render(request, "images/images_list.html", {"images": images})


def images_list_by_category(request, category):
    images = Image.objects.filter(user=request.user, category=category)
    return render(
        request,
        "images/images_list_by_category.html",
        {"category": CATEGORIES.names[category], "images": images},
    )


def upload_image(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            form.save()
            return redirect("images:list")
    return render(request, "images/upload_image.html", {"form": UploadForm})
