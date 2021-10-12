from django.forms import ModelForm

from poseapp.models import Pose


class PoseCreationForm(ModelForm):
    class Meta:
        model = Pose
        fields = ['image']