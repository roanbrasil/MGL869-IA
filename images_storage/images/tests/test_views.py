import os
import pytest
from pathlib import Path
from unittest.mock import patch
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User

from images.models import Image, CATEGORIES
from images.views import classify_images


fixtures_path = Path(os.path.realpath(__file__)).parent / "fixtures"


class MockResponse:
    status_code = 200
    content = b'{"categories": [5,0]}'


def mock_post(*args, **kwargs):
    return MockResponse


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
@patch("images.views.requests.post", mock_post)
def test_can_classify_images():
    """Check that we can add categories to images after model classification."""
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
    images_paths = get_images_paths([image_1, image_2])
    classify_images([image_1, image_2])
    assert image_1.category == CATEGORIES.STREET
    assert image_2.category == CATEGORIES.BUILDINGS
    # Delete images from database
    delete_test_images_from_disk(images_paths)
