# Generated by Django 4.0.6 on 2022-08-18 15:58

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Cloth',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='cloth/')),
                ('upload_date', models.DateField(default=django.utils.timezone.now)),
            ],
        ),
        migrations.CreateModel(
            name='Cloth_data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('shoulder_s', models.IntegerField()),
                ('shoulder_m', models.IntegerField()),
                ('shoulder_l', models.IntegerField()),
                ('shoulder_xl', models.IntegerField()),
                ('shoulder_2l', models.IntegerField()),
                ('chest_s', models.IntegerField()),
                ('chest_m', models.IntegerField()),
                ('chest_l', models.IntegerField()),
                ('chest_xl', models.IntegerField()),
                ('chest_2l', models.IntegerField()),
                ('length_s', models.IntegerField()),
                ('length_m', models.IntegerField()),
                ('length_l', models.IntegerField()),
                ('length_xl', models.IntegerField()),
                ('length_2l', models.IntegerField()),
                ('upload_date', models.DateField(default=django.utils.timezone.now)),
            ],
        ),
    ]
