from django.contrib import admin
from .models import Cloth,Cloth_data,lidardataModel,bodyDataModel

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
    
admin.site.register(Cloth,ClothAdmin)
admin.site.register(Cloth_data,ClothDataAdmin)
admin.site.register(lidardataModel,lidardataAdmin)
admin.site.register(bodyDataModel,bodyDataAdmin)