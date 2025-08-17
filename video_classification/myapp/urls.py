from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Maps root to the home view
]