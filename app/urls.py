from django.urls import path
from . import views

urlpatterns = [
    path('shop_manual',views.shop_manual,name='shop_manual'),
    path('shop_step', views.shop_step),
    path('cloth_img', views.cloth_img, name='cloth_img'),
    path('cloth_data',views.cloth_data,name='cloth_data'),
    path('cloth_preview',views.cloth_preview,name='cloth_preview'),
    path('user_selectCloth',views.user_selectCloth,name='user_selectCloth'),
    path('',views.home,name='home'),
    path('user_manual',views.user_manual,name='user_manual'),
    path('user_showResult',views.user_showResult,name='user_showResult'),
    path('apiTest',views.apiTest,name='apiTest'),
    path('generateImage',views.generateImage,name='generateImage'),
]