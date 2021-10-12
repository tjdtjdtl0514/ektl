from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.urls import reverse
from django.views.generic import CreateView, DetailView

from poseapp.forms import PoseCreationForm
from poseapp.models import Pose


def hello_world(request):
    return render(request, 'base.html')


class PoseCreateView(CreateView):
    model = Pose
    form_class = PoseCreationForm
    template_name = 'poseapp/create.html'

    def get_success_url(self):
        return reverse('poseapp:detail', kwargs={'pk': self.object.pk})


class PoseDetailView(DetailView):
    model = Pose
    context_object_name = 'target_pose'
    template_name = 'poseapp/detail.html'
