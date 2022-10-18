import cv2
import time
import json
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import os
import base64
import glob
import io
import torch
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image,ImageDraw
from skimage import measure, filters
from django.shortcuts import render
from django.core.handlers.wsgi import WSGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, StreamingHttpResponse,JsonResponse
from .forms import ClothesModelForm,ClothesDataModelForm,getEdgeAndLebelForm,generateImageForm
from .models import Cloth,Cloth_data, bodyDataModel,getEdgeAndLebel_data,generateImage_data, lidardataModel, resultImgModel
from app.ganModels.models import create_model
from app import networks
from app.utils.transforms import transform_logits,get_affine_transform
from django.views.decorators.clickjacking import xframe_options_sameorigin
os.environ['CUDA_LAUNCH_BLOCKING']='1'

@csrf_exempt
def apiTest(request:WSGIRequest):
    if request.method =='GET':
        return JsonResponse({'status':'succeed','context':'hello worldd'})

def home(request):
    return render(request,'home.html',locals())
def user_manual(request):
    return render(request,'user_manual.html',locals())

def user_selectCloth(request):
    cloths = Cloth.objects.all()
    time.sleep(1)
    if request.method == "POST":
        poseImg = request.POST.get("poseImg", "")
        keypoints = request.POST.get("keypoints", "")
        shoulderWidth = request.POST.get("shoulderWidth", "")
        chestWidth = request.POST.get("chestWidth", "")
        clothingLength = request.POST.get("clothingLength", "")
        lidardataModel.objects.create(poseImg=poseImg,keypoints=keypoints)
        bodyDataModel.objects.create(shoulderWidth=shoulderWidth,chestWidth=chestWidth,clothingLength=clothingLength)
    
    context = {
        'app': cloths,
        'poseImg': poseImg,
        'keypoints': keypoints,
        'shoulderWidth': shoulderWidth,
        'chestWidth': chestWidth,
        'clothingLength': clothingLength
    }
    return render(request, 'user_selectCloth.html', context)

def cloth_img(request):
    cloths = Cloth.objects.all()
    form = ClothesModelForm()
    if request.method == "POST":
        form = ClothesModelForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            print("save_id:",len(cloths))
            if(len(cloths)>=1):
                cloths=cloths[len(cloths)-1]
            else:
                cloths=cloths[0]
    context = {
        'app': cloths,
        'form': form
    }
    return render(request, 'cloth_img.html', context)
    
def cloth_data(request):
    form = ClothesDataModelForm()
    
    cloths = Cloth.objects.all()
    if(len(cloths)>=1):
        cloths=cloths[len(cloths)-1]
    else:
        cloths=cloths[0]
    print(cloths.id)
    context = {
        'shop':cloths,
        'form': form
    }
    context['form'].fields['image_ID'].initial=cloths.id 
    return render(request,'cloth_data.html',context)

@xframe_options_sameorigin      
def shop_manual(request):
    return render(request,'shop_manual.html',{})

def shop_step(request):
    return render(request,'shop_step.html',{}) 

def xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    aspect_ratio=473 * 1.0 / 473
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale

def box2cs(box):
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h)

dataset_settings = {
    'input_size': [473, 473],
    'num_classes': 20,
    'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
}
num_classes = dataset_settings['num_classes']
input_size = dataset_settings['input_size']
label = dataset_settings['label']
model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
state_dict = torch.load('app/LIP/lip.pth')['state_dict']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

ganModel=create_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
])

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def getEdgeAndLabel(clothImage,humanImage):
    try:
        #edge_part
        img_gray=cv2.cvtColor(clothImage, cv2.COLOR_BGR2GRAY)
        ret ,img_gray = cv2.threshold(img_gray,254, 255, cv2.THRESH_BINARY_INV)
        img_gray = cv2.medianBlur(img_gray, 25)
        img_gray=cv2.resize(img_gray,(192,256),interpolation=cv2.INTER_AREA)
        #label_part
        palette = get_palette(num_classes)
        with torch.no_grad():
            c, s = box2cs([0, 0, 192 - 1, 256 - 1])
            r = 0
            trans = get_affine_transform(c, s, r, input_size)
            input = cv2.warpAffine(
                humanImage,
                trans,
                (int(input_size[1]), int(input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))

            input = transform(input)
            input = input.unsqueeze(0)
            output = model(input)
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            logits_result = transform_logits(upsample_output.data.numpy(), c, s, 192, 256, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            #background
            parsing_result=np.where(parsing_result==0,0,parsing_result)
            #Pants
            parsing_result=np.where(parsing_result==9,8,parsing_result)
            parsing_result=np.where(parsing_result==12,8,parsing_result)
            #hair
            parsing_result=np.where(parsing_result==2,1,parsing_result)
            #face
            parsing_result=np.where(parsing_result==4,12,parsing_result)
            parsing_result=np.where(parsing_result==13,12,parsing_result)
            #upper-cloth
            parsing_result=np.where(parsing_result==5,4,parsing_result)
            parsing_result=np.where(parsing_result==6,4,parsing_result)
            parsing_result=np.where(parsing_result==7,4,parsing_result)
            parsing_result=np.where(parsing_result==10,4,parsing_result)
            #Left-shoe
            parsing_result=np.where(parsing_result==18,5,parsing_result)
            #Right-shoe
            parsing_result=np.where(parsing_result==19,6,parsing_result)
            #Left_leg
            parsing_result=np.where(parsing_result==16,9,parsing_result)
            #Right_leg
            parsing_result=np.where(parsing_result==17,10,parsing_result)
            #Left_arm
            parsing_result=np.where(parsing_result==14,11,parsing_result)
            #Right_arm
            parsing_result=np.where(parsing_result==15,13,parsing_result)
            
    except RuntimeError as re:
        print(re)


    return img_gray,parsing_result
    '''if isShop:
        return render(request, 'cloth_preview.html', context)
    else:
        return render(request, 'user_showResult.html', context)
        '''
    
def changearm(old_label):
    label=old_label
    arm1=torch.FloatTensor((old_label.cpu().numpy()==11).astype(np.int32))
    arm2=torch.FloatTensor((old_label.cpu().numpy()==13).astype(np.int32))
    noise=torch.FloatTensor((old_label.cpu().numpy()==7).astype(np.int32))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label
    
def scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)    


def fflip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
    
    
def get_params(size):
    w, h = size
    new_h = h
    new_w = w
    x = random.randint(0, np.maximum(0, new_w - 512))
    y = random.randint(0, np.maximum(0, new_h - 512))
    #flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}    

def get_transform(params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: scale_width(img, 512, method)))
    osize = [256,192]
    transform_list.append(transforms.Resize(osize, method))  
    transform_list.append(transforms.Lambda(lambda img: fflip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
    
def makePose(pose,params):
    transform_B= get_transform(params)  
    pose_data = np.array(pose)
    pose_data = pose_data.reshape((-1,3))
    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, 256, 192)
    r = 5
    im_pose = Image.new('L', (192, 256))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (192, 256))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i,0]
        pointy = pose_data[i,1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform_B(one_map.convert('RGB'))
        pose_map[i] = one_map[0]
    return pose_map

def reSize(img,keypoints):
    nose=int(keypoints[0])
    img=img[:,(nose-202):(nose+203)]
    img=cv2.resize(img,(192,256),interpolation=cv2.INTER_AREA)
    for i in range(0,42,3):
        keypoints[i],keypoints[i+1]=thansfer(keypoints[i],keypoints[i+1],nose,img)
        
    return img,keypoints 

def thansfer(width,height,nose,img):
    a=(width-(nose-202))*192/405
    b=height*256/540
    return a,b

@csrf_exempt
def generateImage(labelImg,poseImg,colorImg,colorMaskImg,edgeImg,maskImg,keypoints):
    SIZE=320
    NC=14
    # convert and pass the image as base64 string to avoid storing it to DB or filesystem
    labelImage = Image.fromarray(cv2.cvtColor(labelImg.astype(np.uint8),cv2.COLOR_BGR2RGB))
    params=get_params(labelImage.size)
    pose = makePose(keypoints,params)
    pose=pose.unsqueeze(0)
    transform_A = get_transform(params, method=Image.NEAREST, normalize=False)
    transform_B = get_transform(params)
    labelImage=labelImage.convert('L')
    labelTensor = transform_A(labelImage) * 255.0
    labelTensor=labelTensor.unsqueeze(0)
    humanImage = Image.fromarray(cv2.cvtColor(poseImg,cv2.COLOR_BGR2RGB))
    humanTensor = transform_B(humanImage)
    humanTensor=humanTensor.unsqueeze(0)   
    colorImage = Image.fromarray(cv2.cvtColor(colorImg,cv2.COLOR_BGR2RGB))
    colorTensor = transform_B(colorImage)
    colorTensor=colorTensor.unsqueeze(0) 
    colorMaskTensor = transform_A(colorMaskImg)
    edgeImage = Image.fromarray(cv2.cvtColor(edgeImg,cv2.COLOR_BGR2RGB))
    edgeImage=edgeImage.convert('L')
    edgeTensor = transform_A(edgeImage)
    edgeTensor=edgeTensor.unsqueeze(0) 
    maskTensor = transform_A(maskImg)
    maskTensor=maskTensor.unsqueeze(0) 
    try:
        # whether to collect output images
        #save_fake = total_steps % 100 == display_delta
        save_fake = True
        ##add gaussian noise channel
        ## wash the label
        t_mask = torch.FloatTensor((labelTensor.cpu().numpy() == 7).astype(np.float64))
        mask_clothes = torch.FloatTensor((labelTensor.cpu().numpy() == 4).astype(np.int32))
        mask_fore = torch.FloatTensor((labelTensor.cpu().numpy() > 0).astype(np.int32))
        img_fore = humanTensor * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(labelTensor)
        ############## Forward Pass ######################
        losses, fake_image, real_image, input_label,L1_loss,style_loss,clothes_mask,CE_loss,rgb,alpha= ganModel(Variable(labelTensor.cuda()),Variable(edgeTensor.cuda()),Variable(img_fore.cuda()),Variable(mask_clothes.cuda())
                                                                                                    ,Variable(colorTensor.cuda()),Variable(all_clothes_label.cuda()),Variable(humanTensor.cuda()),Variable(pose.cuda()) ,Variable(humanTensor.cuda()) ,Variable(mask_fore.cuda()))
        ### display output images
        generateImage = fake_image.float().cuda()
        generateImage = generateImage[0].squeeze()
        # combine=c[0].squeeze()
        cv_img=(generateImage.permute(1,2,0).detach().cpu().numpy()+1)/2
        rgb=(cv_img*255).astype(np.uint8)
        cv2.imwrite("test.jpg",cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # save to media/bgRemovedImg for showing on html
        resultImg = resultImgModel.objects.all()
        path = 'C:/Users/amy21/Documents/GitHub/MetaMirror_user/media/resultImg_'
        cv2.imwrite(os.path.join(path, 'resultImg_'+ str(len(resultImg)) +'.jpg'), rgb)
        resultImgModel.objects.create(image='resultImg/resultImg'+ str(len(resultImg)) +'.jpg')
    
        rgb=Image.open("test.jpg")
        imgByteArr = io.BytesIO()
        rgb.save(imgByteArr, format='JPEG')
        io_buf = base64.b64encode(imgByteArr.getvalue()).decode('ascii')
        resultImage_uri = 'data:%s;base64,%s' % ('clothImage/jpg', io_buf)
        return resultImage_uri
    except RuntimeError as re:
        print(re)
        
        
    '''
    if isShop:
        return render(request, 'cloth_preview.html', context)
    else:
        return render(request, 'user_showResult.html', context)
    '''
 
def shopGenerateImage(labelImg,poseImg,colorImg,colorMaskImg,edgeImg,maskImg,keypoints):
    
    SIZE=320
    NC=14
    
    # convert and pass the image as base64 string to avoid storing it to DB or filesystem
    labelImage = Image.fromarray(cv2.cvtColor(labelImg.astype(np.uint8),cv2.COLOR_BGR2RGB))
    params=get_params(labelImage.size)
    pose = keypoints
    pose = makePose(pose,params)
    pose=pose.unsqueeze(0)
    transform_A = get_transform(params, method=Image.NEAREST, normalize=False)
    transform_B = get_transform(params)
    labelImage=labelImage.convert('L')
    labelTensor = transform_A(labelImage) * 255.0
    labelTensor=labelTensor.unsqueeze(0)
    humanImage = Image.fromarray(cv2.cvtColor(poseImg,cv2.COLOR_BGR2RGB))
    humanTensor = transform_B(humanImage)
    humanTensor=humanTensor.unsqueeze(0)   
    colorImage = Image.fromarray(cv2.cvtColor(colorImg,cv2.COLOR_BGR2RGB))
    colorTensor = transform_B(colorImage)
    colorTensor=colorTensor.unsqueeze(0) 
    colorMaskTensor = transform_A(colorMaskImg)
    
    edgeImage = Image.fromarray(cv2.cvtColor(edgeImg,cv2.COLOR_BGR2RGB))
    edgeImage=edgeImage.convert('L')
    edgeTensor = transform_A(edgeImage)
    edgeTensor=edgeTensor.unsqueeze(0) 
    maskTensor = transform_A(maskImg)
    maskTensor=maskTensor.unsqueeze(0) 
    
    try:
        # whether to collect output images
        #save_fake = total_steps % 100 == display_delta
        save_fake = True
        ##add gaussian noise channel
        ## wash the label
        t_mask = torch.FloatTensor((labelTensor.cpu().numpy() == 7).astype(np.float64))
        mask_clothes = torch.FloatTensor((labelTensor.cpu().numpy() == 4).astype(np.int32))
        mask_fore = torch.FloatTensor((labelTensor.cpu().numpy() > 0).astype(np.int32))
        img_fore = humanTensor * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(labelTensor)
        ############## Forward Pass ######################
        losses, fake_image, real_image, input_label,L1_loss,style_loss,clothes_mask,CE_loss,rgb,alpha= ganModel(Variable(labelTensor.cuda()),Variable(edgeTensor.cuda()),Variable(img_fore.cuda()),Variable(mask_clothes.cuda())
                                                                                                    ,Variable(colorTensor.cuda()),Variable(all_clothes_label.cuda()),Variable(humanTensor.cuda()),Variable(pose.cuda()) ,Variable(humanTensor.cuda()) ,Variable(mask_fore.cuda()))
        ### display output images
        generateImage = fake_image.float().cuda()
        generateImage = generateImage[0].squeeze()
        # combine=c[0].squeeze()
        cv_img=(generateImage.permute(1,2,0).detach().cpu().numpy()+1)/2
        rgb=(cv_img*255).astype(np.uint8)
        cv2.imwrite("test.jpg",cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        rgb=Image.open("test.jpg")
        imgByteArr = io.BytesIO()
        rgb.save(imgByteArr, format='JPEG')
        io_buf = base64.b64encode(imgByteArr.getvalue()).decode('ascii')
        resultImage_uri = 'data:%s;base64,%s' % ('clothImage/jpg', io_buf)
        return resultImage_uri
    except RuntimeError as re:
        print(re) 
 
    
def user_showResult(request):
    bodyDataName = ["肩寬","胸寬","身長"]
    size_str = ""
    size_cnt = []
    size_result = ""
    
    lidardata = lidardataModel.objects.all()
    if(len(lidardata)>=1):
        lidardata=lidardata[len(lidardata)-1]
    else:
        lidardata=lidardata[0]
    
    bodyData = bodyDataModel.objects.all()
    if(len(bodyData)>=1):
        bodyData=bodyData[len(bodyData)-1]
    else:
        bodyData=bodyData[0]
    bodyData = [float(bodyData.shoulderWidth),float(bodyData.chestWidth),float(bodyData.clothingLength)]
    
        
    #get user selection of cloth image and data
    cloth = None
    cloth_data = None
    if request.method == "POST":
        cloth=Cloth.objects.get(id=request.POST['cloth'])
        cloth_data=Cloth_data.objects.get(image_ID=request.POST['cloth'])
        poseImg=lidardata.poseImg
        keypoints=lidardata.keypoints
        keypoints=keypoints[1:-1]
        keypoints = keypoints.split(",")
        keypoints = list(map(float, keypoints))
        colorImg=str(cloth.image)
        io_buf = base64.b64decode(poseImg)
        poseImg = np.frombuffer(io_buf, dtype=np.uint8)
        poseImg=cv2.imdecode(poseImg,cv2.IMREAD_COLOR)
        colorImg=cv2.imread("media/"+colorImg)
        poseImg,keypoints=reSize(poseImg,keypoints)
        
        ret = str(random.randint(0, 9999)).zfill(5)
        maskImg=Image.open('app/test_mask/'+ret+'.png').convert('L')
        colorMaskImg=Image.open('app/test_colormask/'+ret+'_test.png').convert('L')
        edgeImg,labelImg=getEdgeAndLabel(colorImg,poseImg)
    
    # size chart, need to import from database
    chart = [[cloth_data.shoulder_s, cloth_data.shoulder_m, cloth_data.shoulder_l, cloth_data.shoulder_xl, cloth_data.shoulder_2l],
            [cloth_data.chest_s, cloth_data.chest_m, cloth_data.chest_l, cloth_data.chest_xl, cloth_data.chest_2l],
            [cloth_data.length_s, cloth_data.length_m, cloth_data.length_l, cloth_data.length_xl, cloth_data.length_2l]]

    # compare with size chart
    for i in range(0, 3):
        bodyData[i] = np.round(bodyData[i],2)
        if bodyData[i] <= chart[i][0]:
            size_str += "S"
        elif bodyData[i] >= chart[i][0] and bodyData[i] <= chart[i][1]:
            size_str += "M"
        elif bodyData[i] >= chart[i][1] and bodyData[i] <= chart[i][2]:
            size_str += "L"
        elif bodyData[i] >= chart[i][2] and bodyData[i] <= chart[i][3]:
            size_str += "XL"
        else:
            size_str += "2XL"

    # descending order, because of index()
    size_cnt.append(size_str.count("2XL"))
    size_cnt.append(size_str.count("XL"))
    size_cnt.append(size_str.count("L"))
    size_cnt.append(size_str.count("M"))
    size_cnt.append(size_str.count("S"))
    print(size_str)
    print(size_cnt)

    recommend_size = size_cnt.index(max(size_cnt))
    if recommend_size == 0:
        size_result = "The fit size is 2XL and the loose size is 3XL"
        print("INFO: The fit size is 2XL and the loose size is 3XL")
    elif recommend_size == 1:
        size_result = "The fit size is XL and the loose size is 2XL"
        print("INFO: The fit size is XL and the loose size is 2XL")
    elif recommend_size == 2:
        size_result = "The fit size is L and the loose size is XL"
        print("INFO: The fit size is L and the loose size is XL")
    elif recommend_size == 3:
        size_result = "The fit size is M and the loose size is L"
        print("INFO: The fit size is M and the loose size is L")
    else:
        size_result = "The fit size is S and the loose size is M"
        print("INFO: The fit size is S and the loose size is M")

    bodyDataList = zip(bodyDataName , bodyData)
        
    #try on
    resultImage_uri=generateImage(labelImg, poseImg, colorImg, colorMaskImg, edgeImg, maskImg, keypoints)
        
    context = {
        'bodyDataList': bodyDataList,
        'size_result': size_result,
        'resultImage':resultImage_uri,
        'selectedcloth_img':cloth,
        'text':cloth_data
    }
    
    return render(request,'user_showResult.html', context)

    
def cloth_preview(request):
    form = ClothesDataModelForm()
    cloths = Cloth.objects.all()
    if(len(cloths)>=1):
        cloths=cloths[len(cloths)-1]
    else:
        cloths=cloths[0]
    print(cloths.image)

    model_img = cv2.imread('020000_0.jpg')
    cloth_img = cv2.imread('media/'+str(cloths.image))
    maskImg=Image.open('00000.png').convert('L')
    colorMaskImg=Image.open('00000_test.png').convert('L')
    p_keypoints=[479, 142, 1.3290001153945923, 
                 472, 221, 1.4182500839233398,
                 401, 218, 1.4267500638961792, 
                 360, 332, 1.4455000162124634, 
                 304, 452, 1.4095001220703125, 
                 543, 224, 1.4207500219345093, 
                 569, 342, 1.4622501134872437, 
                 610, 451, 1.4277501106262207, 
                 423, 427, 1.3007500171661377, 
                 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 
                 505, 423, 1.311000108718872, 
                 0.0, 0.0, 0.0, 
                 0.0, 0.0, 0.0, 
                 464, 126, 1.3532500267028809, 
                 496, 127, 1.3475000858306885, 
                 450, 136, 1.3632500171661377, 
                 509, 137, 1.3595000505447388]
    model_img,p_keypoints=reSize(model_img,p_keypoints)

    if request.method == "POST":
        form = ClothesDataModelForm(request.POST)
        #print(request.POST['image_ID'])
        if form.is_valid():
            cloth_info=form.cleaned_data
            cloth_info['image_ID']=cloths.id
            print("cloth_info:",cloth_info)
            Cloth_data.objects.create(**cloth_info)
            img_gray,parsing_result=getEdgeAndLabel(cloth_img,model_img)
            resultImage=shopGenerateImage(parsing_result, model_img, cloth_img, colorMaskImg, img_gray, maskImg, p_keypoints)
            print("result_img:",resultImage)

    context = {
        'app': cloths,
        'text': form,
        'resultImage':resultImage
    }
    context['text'].fields['image_ID'].initial=cloths.id    
    return render(request,'cloth_preview.html',context)