import os
import pytest
from pathlib import Path
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils.datastructures import MultiValueDict
from django.contrib.auth.models import User

from images.models import Image
from images.forms import UploadForm, DeleteFormSet

fixtures_path = Path(os.path.realpath(__file__)).parent / "fixtures"


def get_images_paths(images: list[Image]) -> list[str]:
    paths = []
    for image in images:
        paths.append(image.src.path)
    return paths


def delete_test_images_from_disk(images_paths: list[str]) -> None:
    for image_path in images_paths:
        if os.path.exists(image_path):
            os.remove(image_path)


@pytest.mark.django_db()
def test_can_add_multiple_images():
    """Test that multiple images can be added."""
    user = User.objects.create_user(username="testuser")
    files = [
        SimpleUploadedFile("street.jpg", (fixtures_path / "street.jpg").read_bytes()),
        SimpleUploadedFile(
            "buildings.jpg", (fixtures_path / "buildings.jpg").read_bytes()
        ),
    ]
    form = UploadForm(
        data={},
        files=MultiValueDict({"images": files}),
        user=user,
    )
    assert form.is_valid()
    form.save()
    images = Image.objects.filter(user=user)
    assert images.count() == 2
    # Delete images from database
    images_paths = get_images_paths(images)
    delete_test_images_from_disk(images_paths)


@pytest.mark.django_db()
def test_can_delete_multiple_images():
    """Test that multiple images can be deleted."""
    user = User.objects.create_user(username="testuser")
    image_1 = Image.objects.create(
        user=user,
        src=SimpleUploadedFile(
            "street.jpg", (fixtures_path / "street.jpg").read_bytes()
        ),
    )
    image_2 = Image.objects.create(
        user=user,
        src=SimpleUploadedFile(
            "buildings.jpg", (fixtures_path / "buildings.jpg").read_bytes()
        ),
    )
    images = Image.objects.filter(user=user)
    data = {
        "form-TOTAL_FORMS": "2",
        "form-INITIAL_FORMS": "2",
        "form-0-id": f"{image_1.pk}",
        "form-0-user": "1",
        "form-0-process": "2",
        "form-0-delete": "on",
        "form-1-id": f"{image_2.pk}",
        "form-1-user": "1",
        "form-1-process": "2",
        "form-1-delete": "on",
    }
    images_paths = get_images_paths(images)
    formset = DeleteFormSet(data=data, queryset=images)
    assert formset.is_valid()
    formset.save()
    images = Image.objects.filter(user=user)
    assert images.count() == 0
    # Delete images from database
    delete_test_images_from_disk(images_paths)
