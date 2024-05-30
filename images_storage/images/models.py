from django.contrib.auth.models import User
from django.db import models
from django_cleanup import cleanup


class CATEGORIES(models.IntegerChoices):
    BUILDINGS = 0, "buildings"
    FOREST = 1, "forest"
    GLACIER = 2, "glacier"
    MOUNTAIN = 3, "mountain"
    SEA = 4, "sea"
    STREET = 5, "street"


class PROCESSES(models.IntegerChoices):
    TRAINING = 0, "training"
    VALIDATION = 1, "validation"
    TEST = 2, "test"


@cleanup.select
class Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    src = models.ImageField(upload_to='images_repository', verbose_name="Image")
    category = models.IntegerField(null=True, blank=True, verbose_name="Category", choices=CATEGORIES.choices)
    process = models.IntegerField(verbose_name="Process", choices=PROCESSES.choices, default=PROCESSES.TEST)
