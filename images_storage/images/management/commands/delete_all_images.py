from pathlib import Path
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from images.models import Image, PROCESSES
from django.conf import settings


class Command(BaseCommand):
    help = "Import training and validation images"

    def handle(self, *args, **options):
        Image.objects.filter(process=PROCESSES.TRAINING).delete()
        Image.objects.filter(process=PROCESSES.VALIDATION).delete()
