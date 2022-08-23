from django.urls import path,re_path
from . import views

urlpatterns = [
    path('shop_manual',views.shop_manual,name='shop_manual'),
    path('cloth_img', views.cloth_img, name='cloth_img'),
    path('cloth_data',views.cloth_data,name='cloth_data'),
    path('cloth_preview',views.cloth_preview,name='cloth_preview'),
    path('user_selectCloth',views.user_selectCloth,name='user_selectCloth'),
    path('',views.home,name='home'),
]