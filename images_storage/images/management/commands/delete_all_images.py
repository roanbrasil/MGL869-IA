from django.core.management.base import BaseCommand
from images.models import Image, PROCESSES


class Command(BaseCommand):
    help = "Import training and validation images"

    def handle(self, *args, **options):
        Image.objects.filter(process=PROCESSES.TRAINING).delete()
        Image.objects.filter(process=PROCESSES.VALIDATION).delete()
