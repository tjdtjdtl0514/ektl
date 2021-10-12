from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from poseapp.views import hello_world, PoseCreateView, PoseDetailView

app_name = "poseapp"

urlpatterns = [
    path('hello_world/', hello_world, name='hello_world'),
    path('create/', PoseCreateView.as_view(), name='create'),
    path('detail/<int:pk>', PoseDetailView.as_view(), name='detail'),

]

