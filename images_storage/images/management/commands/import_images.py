from pathlib import Path
import shutil
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from images.models import Image, CATEGORIES, PROCESSES


class Command(BaseCommand):
    help = "Import training and validation images"

    def add_arguments(self, parser):
        parser.add_argument("--images_src", action="store", dest="images_src", type=str)

    def handle(self, *args, **options):
        admin_user = User.objects.get(username="admin")
        images_src = Path(options.get("images_src"))
        training_images_src = images_src / "seg_train"
        validation_images_src = images_src / "seg_test"
        for i in CATEGORIES.values:
            for image_src in (training_images_src / CATEGORIES(i).label).iterdir():
                image = Image.objects.create(
                    category=CATEGORIES(i).value,
                    user=admin_user,
                    process=PROCESSES.TRAINING,
                )
                image.src.save(image_src.name, ContentFile(image_src.read_bytes()))
            for image_src in (validation_images_src / CATEGORIES(i).label).iterdir():
                image = Image.objects.create(
                    category=CATEGORIES(i).value,
                    user=admin_user,
                    process=PROCESSES.VALIDATION,
                )
                image.src.save(image_src.name, ContentFile(image_src.read_bytes()))
