from django.urls import path
from . import views

urlpatterns = [
    path('shop_manual',views.shop_manual,name='shop_manual'),
    path('cloth_img', views.cloth_img, name='cloth_img'),
    path('cloth_data',views.cloth_data,name='cloth_data'),
    path('cloth_preview',views.cloth_preview,name='cloth_preview'),
    path('home',views.home,name='home'),
]