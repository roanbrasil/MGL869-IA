"""
    BORROWED CODE :
    The following lines are based on the code from:
    https://github.com/gitdagray/django-course/blob/main/lesson12/myproject/users/views.py
    We have borrowed the logic for the user's registering, login, and logout.
"""

from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    path('register/', views.register_view, name="register"),
    path('login/', views.login_view, name="login"),
    path('logout/', views.logout_view, name="logout"),
]

"""END OF BORROWED CODE"""
