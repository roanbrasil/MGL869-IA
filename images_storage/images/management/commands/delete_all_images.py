from django.core.management.base import BaseCommand
from images.models import Image


class Command(BaseCommand):
    help = "Delete all images from the database."

    def handle(self, *args, **options):
        Image.objects.all().delete()
