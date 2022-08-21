from django.contrib import admin
from .models import Cloth,Cloth_data

# Register your models here.
class ClothAdmin(admin.ModelAdmin):
    list_display = ('image', 'upload_date')
class ClothDataAdmin(admin.ModelAdmin):
    list_display = ('shoulder_s','shoulder_m','shoulder_l','shoulder_xl','shoulder_2l',
        'chest_s','chest_m','chest_l','chest_xl','chest_2l',
        'length_s','length_m','length_l','length_xl','length_2l')

admin.site.register(Cloth,ClothAdmin)
admin.site.register(Cloth_data,ClothDataAdmin)
