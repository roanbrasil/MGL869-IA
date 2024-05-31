from django.shortcuts import render, redirect

from images.forms import DeleteFormSet, UploadForm
from images.models import Image, CATEGORIES, PROCESSES
from model.classification import classify


def images_list(request):
    images = Image.objects.filter(user=request.user, process=PROCESSES.TEST)
    return render(request, "images/images_list.html", {"images": images})


def images_list_by_category(request, category):
    images = Image.objects.filter(user=request.user, category=category, process=PROCESSES.TEST)
    return render(
        request,
        "images/images_list_by_category.html",
        {"category": CATEGORIES.names[category], "images": images},
    )


def upload_images(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            form.save()
            # Classify the new uploaded images
            uploaded_images = Image.objects.filter(
                user=request.user, process=PROCESSES.TEST, category=None
            )
            classify(uploaded_images)
            return redirect("images:list")
    return render(request, "images/upload_images.html", {"form": UploadForm})


def delete_images(request):
    images = Image.objects.filter(user=request.user)
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
