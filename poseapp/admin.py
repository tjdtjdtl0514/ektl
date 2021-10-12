from django.contrib import admin

# Register your models here.
from poseapp.models import Pose


@admin.register(Pose)
class PoseAdmin(admin.ModelAdmin):
    pass