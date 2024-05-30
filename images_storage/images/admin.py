from django.contrib import admin
from .models import Image


class ImageAdmin(admin.ModelAdmin):
    readonly_fields = ["id", "user"]
    list_display = ["id", "user", "category", "process"]
    list_filter = ["category", "process"]


admin.site.register(Image, ImageAdmin)
