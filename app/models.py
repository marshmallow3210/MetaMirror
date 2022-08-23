from django.db import models
from django.utils import timezone
# Create your models here.
class Cloth(models.Model):
    #衣服圖片
    image = models.ImageField(upload_to='cloth/', blank=False, null=False)
    upload_date = models.DateField(default=timezone.now)

class Cloth_data(models.Model):
    #衣服資訊
    #image_name = models.TextField()
    shoulder_s=models.IntegerField()
    shoulder_m=models.IntegerField()
    shoulder_l=models.IntegerField()
    shoulder_xl=models.IntegerField()
    shoulder_2l=models.IntegerField()

    chest_s=models.IntegerField()
    chest_m=models.IntegerField()
    chest_l=models.IntegerField()
    chest_xl=models.IntegerField()
    chest_2l=models.IntegerField()

    length_s=models.IntegerField()
    length_m=models.IntegerField()
    length_l=models.IntegerField()
    length_xl=models.IntegerField()
    length_2l=models.IntegerField()
    upload_date = models.DateField(default=timezone.now)
    
class getEdgeAndLebel_data(models.Model):
    #是否為商店端
    isShop=models.BooleanField()
    #衣服圖片
    clothImage = models.ImageField()
    #人物圖片
    humanImage = models.ImageField()
    
    
class generateImage_data(models.Model):
    #是否為商店端
    isShop=models.BooleanField()
    #標籤
    label = models.ImageField()
    #人物圖片
    image = models.ImageField()
    #衣服圖片
    color = models.ImageField()
    #衣服雜訊
    colorMask = models.ImageField()
    #衣服輪廓
    edge =  models.ImageField()
    #人物雜訊
    mask =  models.ImageField()
    #人物關鍵點
    pose = models.FloatField()