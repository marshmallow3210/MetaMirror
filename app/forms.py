from django import forms
from .models import Cloth,Cloth_data,getEdgeAndLebel_data,generateImage_data

class ClothseModelForm(forms.ModelForm):
    class Meta:
        model = Cloth
        fields=('image',)
        widgets={
            'image': forms.FileInput(attrs={'class': 'form-control-file'})
        }

class ClothseDataModelForm(forms.ModelForm):
    
    class Meta:
        model = Cloth_data
        fields=('shoulder_s','shoulder_m','shoulder_l','shoulder_xl','shoulder_2l',
        'chest_s','chest_m','chest_l','chest_xl','chest_2l',
        'length_s','length_m','length_l','length_xl','length_2l')
        widgets={
            #'image_name': forms.TextInput(attrs={'class': 'form-control'}),
            'shoulder_s':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'shoulder_m':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'shoulder_l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'shoulder_xl':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'shoulder_2l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            
            'chest_s':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'chest_m':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'chest_l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'chest_xl':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'chest_2l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            
            'length_s':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'length_m':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'length_l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'length_xl':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
            'length_2l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm'}),
        }
        
        
class getEdgeAndLebelForm(forms.ModelForm):
     class Meta:
            model = getEdgeAndLebel_data
            fields=('isShop','clothImage','humanImage')
            widgets={
            'isShop':forms.NumberInput(attrs={'class': 'form-control'}),
            'clothImage': forms.FileInput(attrs={'class': 'form-control-file'}),
            'humanImage': forms.FileInput(attrs={'class': 'form-control-file'})
         }
            
            

class generateImageForm(forms.ModelForm):
     class Meta:
            model = generateImage_data
            fields=('isShop','label','image','color','colorMask','edge','mask','pose')
            widgets={
            'isShop':forms.NumberInput(attrs={'class': 'form-control'}),
            'label': forms.FileInput(attrs={'class': 'form-control-file'}),
            'image': forms.FileInput(attrs={'class': 'form-control-file'}),
            'color': forms.FileInput(attrs={'class': 'form-control-file'}),
            'colorMask': forms.FileInput(attrs={'class': 'form-control-file'}),
            'edge': forms.FileInput(attrs={'class': 'form-control-file'}),
            'mask': forms.FileInput(attrs={'class': 'form-control-file'}),
            'pose':forms.NumberInput(attrs={'class': 'form-control'}),
         }
