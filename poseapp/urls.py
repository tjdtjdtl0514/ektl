from django.urls import path

from poseapp.views import hello_world

app_name = "poseapp"

urlpatterns = [
    path('hello_world/', hello_world, name='hello_world')

]