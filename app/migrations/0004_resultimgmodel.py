# Generated by Django 4.1.1 on 2022-10-18 02:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_bodydatamodel_lidardatamodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='resultImgModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='resultImg/')),
            ],
        ),
    ]
