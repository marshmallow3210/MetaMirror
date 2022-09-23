from django import forms
from .models import Cloth,Cloth_data,getEdgeAndLebel_data,generateImage_data,KeypointsModel

class ClothesModelForm(forms.ModelForm):
    class Meta:
        model = Cloth
        fields=('image',)
        widgets={
            'image': forms.FileInput(attrs={'class': 'form-control-file'})
        }
        
class KeypointsModelForm(forms.ModelForm):
    class Meta:
        model = KeypointsModel
        fields=('image',)
        widgets={
            'image': forms.FileInput(attrs={'class': 'form-control-file'})
        }
        
class ClothesDataModelForm(forms.ModelForm):
    
    class Meta:
        model = Cloth_data
        fields=('image_ID','shoulder_s','shoulder_m','shoulder_l','shoulder_xl','shoulder_2l',
        'chest_s','chest_m','chest_l','chest_xl','chest_2l',
        'length_s','length_m','length_l','length_xl','length_2l')
        widgets={
            'image_ID':forms.NumberInput(attrs={'type':"hidden",'value':"{{shop.id}}"}),
            'shoulder_s':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'shoulder_m':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'shoulder_l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'shoulder_xl':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'shoulder_2l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            
            'chest_s':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'chest_m':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'chest_l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'chest_xl':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'chest_2l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            
            'length_s':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'length_m':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'length_l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'length_xl':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
            'length_2l':forms.NumberInput(attrs={'class': 'form-control','placeholder':'cm','min':"1", 'max':"100"}),
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
