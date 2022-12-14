# Generated by Django 4.0.6 on 2022-09-07 13:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='generateImage_data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('isShop', models.BooleanField()),
                ('label', models.ImageField(upload_to='')),
                ('image', models.ImageField(upload_to='')),
                ('color', models.ImageField(upload_to='')),
                ('colorMask', models.ImageField(upload_to='')),
                ('edge', models.ImageField(upload_to='')),
                ('mask', models.ImageField(upload_to='')),
                ('pose', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='getEdgeAndLebel_data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('isShop', models.BooleanField()),
                ('clothImage', models.ImageField(upload_to='')),
                ('humanImage', models.ImageField(upload_to='')),
            ],
        ),
        migrations.AddField(
            model_name='cloth_data',
            name='image_ID',
            field=models.IntegerField(default=0),
        ),
    ]
