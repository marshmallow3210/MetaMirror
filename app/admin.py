from django.contrib import admin
from .models import Cloth,Cloth_data,lidardataModel,bodyDataModel, originalPoseImgModel,resultImgModel

# Register your models here.
class ClothAdmin(admin.ModelAdmin):
    list_display = ('image', 'upload_date')
    
class ClothDataAdmin(admin.ModelAdmin):
    list_display = ('image_ID','shoulder_s','shoulder_m','shoulder_l','shoulder_xl','shoulder_2l',
        'chest_s','chest_m','chest_l','chest_xl','chest_2l',
        'length_s','length_m','length_l','length_xl','length_2l')

class lidardataAdmin(admin.ModelAdmin):
    list_display = ('poseImg','keypoints')
    
class bodyDataAdmin(admin.ModelAdmin):
    list_display = ('shoulderWidth','chestWidth','clothingLength')
    
class resultImgAdmin(admin.ModelAdmin):
    list_display = ('image',)
    
class originalPoseImgAdmin(admin.ModelAdmin):
    list_display = ('originalPoseImg',)
    
admin.site.register(Cloth,ClothAdmin)
admin.site.register(Cloth_data,ClothDataAdmin)
admin.site.register(lidardataModel,lidardataAdmin)
admin.site.register(bodyDataModel,bodyDataAdmin)
admin.site.register(resultImgModel,resultImgAdmin)
admin.site.register(originalPoseImgModel,originalPoseImgAdmin)