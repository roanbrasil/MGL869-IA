from django.urls import path
from . import views

app_name = 'images'

urlpatterns = [
    path('list/', views.images_list, name="list"),
    path('list_by_category/<int:category>', views.images_list_by_category, name="list_by_category"),
    path('upload/', views.upload_image, name="upload"),
]
