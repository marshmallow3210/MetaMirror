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
from .models import Cloth,Cloth_data,getEdgeAndLebel_data,generateImage_data
from app.ganModels.models import create_model
from app import networks
from app.utils.transforms import transform_logits,get_affine_transform
os.environ['CUDA_LAUNCH_BLOCKING']='1'

@csrf_exempt
def apiTest(request:WSGIRequest):
    if request.method =='GET':
        return JsonResponse({'status':'succeed','context':'hello worldd'})

def home(request):
    return render(request,'home.html',locals())
def user_manual(request):
    return render(request,'user_manual.html',locals())
'''
def runLidar():    
    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Create a config and configure the pipeline to stream different resolutions of color and depth streams
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)    
        width = 960
        height = 540
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        width = 640
        height = 480

    # keypoints detection from mediapipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # We will be removing the background of objects more than clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.6 # meters
    clipping_distance = clipping_distance_in_meters / depth_scale
    
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    nosePos = [0, 0, 0, 0]
    eyesPos = [0, 0, 0, 0]
    earPos = [0, 0, 0, 0]
    shoulderPos = [0, 0, 0, 0] # leftPosX, leftPosY, rightPosX, rightPosY
    hipPos = [0, 0, 0, 0] 
    elbowPos = [0, 0, 0, 0] 
    wristPos = [0, 0, 0, 0] 
    
    con = 10
    print('start')
    
    while True:
        con -= 1
        
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 解碼圖片
        decode_frames = cv2.imencode('.jpeg', color_image)
        decode_array = decode_frames[1]
        # 轉換成byte，存在迭代器中
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + decode_array.tobytes() + b'\r\n')		
        print('decode_array type is', type(decode_array))
        
        results = holistic.process(color_image)
        
        if  con < 2 and con > 0 and results.pose_landmarks:
            nosePos[0] = nosePos[0] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x*width)
            nosePos[1] = nosePos[1] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y*height)
                
            eyesPos[0] = eyesPos[0] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x*width)
            eyesPos[1] = eyesPos[1] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y*height)
            eyesPos[2] = eyesPos[2] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x*width)
            eyesPos[3] = eyesPos[3] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y*height)
                
            earPos[0] = earPos[0] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x*width)
            earPos[1] = earPos[1] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y*height)
            earPos[2] = earPos[2] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x*width)
            earPos[3] = earPos[3] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y*height)
                
            shoulderPos[0] = shoulderPos[0] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x*width)
            shoulderPos[1] = shoulderPos[1] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y*height)
            shoulderPos[2] = shoulderPos[2] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x*width)
            shoulderPos[3] = shoulderPos[3] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y*height)
            
            hipPos[0] = hipPos[0] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x*width)
            hipPos[1] = hipPos[1] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y*height)
            hipPos[2] = hipPos[2] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x*width)
            hipPos[3] = hipPos[3] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y*height)
            
            elbowPos[0] = elbowPos[0] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x*width)
            elbowPos[1] = elbowPos[1] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y*height)
            elbowPos[2] = elbowPos[2] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x*width)
            elbowPos[3] = elbowPos[3] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y*height)
            
            wristPos[0] = wristPos[0] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x*width)
            wristPos[1] = wristPos[1] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y*height)
            wristPos[2] = wristPos[2] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x*width)
            wristPos[3] = wristPos[3] + (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y*height)
        
        
        print(con)
        if con <= 0:
            pipeline.stop()
            cv2.imwrite('keypoints.jpg', color_image)
            print('end')
            break
        time.sleep(0.2)
        
    # Intrinsics & Extrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 255
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    images = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB) # 顏色標記
    images = cv2.resize(images, (960, 540))
    images = cv2.bilateralFilter(images,9,75,75) # 雙向濾波
    kernel = np.ones((3, 3), np.uint8)
    images = cv2.morphologyEx(images, cv2.MORPH_CLOSE, kernel) # closing

    # Labelling connected components
    # 處理頭髮
    n = 1
    img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    img = filters.gaussian(img, sigma = 3 / (4. * n))
    labels = measure.label(img, background=0)
    labels = np.where((labels > 1), 0, labels)
    labels_3d = np.dstack((labels,labels,labels)) # 3 channels
    
    # Remove background
    bg_removed = np.where((labels_3d > 0), 255, color_image)
    images = cv2.resize(bg_removed, (960, 540))
    images = cv2.bilateralFilter(images,9,75,75) # 雙向濾波
    kernel = np.ones((3, 3), np.uint8)
    images = cv2.morphologyEx(images, cv2.MORPH_CLOSE, kernel) # closing
    cv2.imwrite('keypoints_bg_removed.jpg', images)
    
    
    # get keypoints' 3d coordinate
    # nose
    for i in range(0, 4): # 0 to 3
        nosePos[i] = int(nosePos[i] // 1)
    nose_xy = [nosePos[0], nosePos[1]] 
    nose_depth = aligned_depth_frame.get_distance(int(nosePos[0]), int(nosePos[1]))
    print("INFO: The position of nose is", nose_xy, "px,", nose_depth, "m")
    
    # eyes
    for i in range(0, 4): # 0 to 3
        eyesPos[i] = int(eyesPos[i] // 1)
    eye_xyL = [eyesPos[2], eyesPos[3]] 
    eye_xyR = [eyesPos[0], eyesPos[1]] 
    eye_depthL = aligned_depth_frame.get_distance(int(eyesPos[0]), int(eyesPos[1]))
    eye_depthR = aligned_depth_frame.get_distance(int(eyesPos[2]), int(eyesPos[3]))
    print("INFO: The position of left eye is", eye_xyL, "px,", eye_depthL, "m")
    print("INFO: The position of right eye is", eye_xyR, "px,", eye_depthR, "m")
    
    # ear
    for i in range(0, 4): # 0 to 3
        earPos[i] = int(earPos[i] // 1)
    ear_xyL = [earPos[2], earPos[3]] 
    ear_xyR = [earPos[0], earPos[1]] 
    ear_depthL = aligned_depth_frame.get_distance(int(earPos[0]), int(earPos[1]))
    ear_depthR = aligned_depth_frame.get_distance(int(earPos[2]), int(earPos[3]))
    print("INFO: The position of left ear is", ear_xyL, "px,", ear_depthL, "m")
    print("INFO: The position of right ear is", ear_xyR, "px,", ear_depthR, "m")
    
    # shoulder
    for i in range(0, 4): # 0 to 3
        shoulderPos[i] = int(shoulderPos[i] // 1) # average position
    shoulder_xyL = [shoulderPos[2], shoulderPos[3]] 
    shoulder_xyR = [shoulderPos[0], shoulderPos[1]] 
    shoulder_xyM = [int((shoulderPos[0]+shoulderPos[2])/2), int((shoulderPos[1]+shoulderPos[3])/2)]
    shoulder_depthL = aligned_depth_frame.get_distance(int(shoulderPos[0]), int(shoulderPos[1]))
    shoulder_depthR = aligned_depth_frame.get_distance(int(shoulderPos[2]), int(shoulderPos[3])) 
    shoulder_depthM = aligned_depth_frame.get_distance(int(shoulder_xyM[0]), int(shoulder_xyM[1]))
    shoulderxyzL = rs.rs2_deproject_pixel_to_point(depth_intrin, shoulder_xyL, shoulder_depthL)
    shoulderxyzR = rs.rs2_deproject_pixel_to_point(depth_intrin, shoulder_xyR, shoulder_depthR)
    print("INFO: The position of left shoulder is", shoulder_xyL, "px,", shoulder_depthL, "m")
    print("INFO: The position of right shoulder is", shoulder_xyR, "px,", shoulder_depthR, "m")
    print("INFO: The position of medium shoulder is", shoulder_xyM, "px,", shoulder_depthM, "m")
    
    # elbow
    for i in range(0, 4): # 0 to 3
        elbowPos[i] = int(elbowPos[i] // 1)
    elbow_xyL = [elbowPos[2], elbowPos[3]] 
    elbow_xyR = [elbowPos[0], elbowPos[1]] 
    elbow_depthL = aligned_depth_frame.get_distance(int(elbowPos[0]), int(elbowPos[1]))
    elbow_depthR = aligned_depth_frame.get_distance(int(elbowPos[2]), int(elbowPos[3]))
    elbowxyzL = rs.rs2_deproject_pixel_to_point(depth_intrin, elbow_xyL, elbow_depthL)
    elbowxyzR = rs.rs2_deproject_pixel_to_point(depth_intrin, elbow_xyR, elbow_depthR)
    print("INFO: The position of left elbow is", elbow_xyL, "px,", elbow_depthL, "m")
    print("INFO: The position of right elbow is", elbow_xyR, "px,", elbow_depthR, "m")
    
    # wrist
    for i in range(0, 4): # 0 to 3
        wristPos[i] = int(wristPos[i] // 1)
    wrist_xyL = [wristPos[2], wristPos[3]] 
    wrist_xyR = [wristPos[0], wristPos[1]] 
    wrist_depthL = aligned_depth_frame.get_distance(int(wristPos[0]), int(wristPos[1]))
    wrist_depthR = aligned_depth_frame.get_distance(int(wristPos[2]), int(wristPos[3]))
    wristxyzL = rs.rs2_deproject_pixel_to_point(depth_intrin, wrist_xyL, wrist_depthL)
    wristxyzR = rs.rs2_deproject_pixel_to_point(depth_intrin, wrist_xyR, wrist_depthR)
    print("INFO: The position of left wrist is", wrist_xyL, "px,", wrist_depthL, "m")
    print("INFO: The position of right wrist is", wrist_xyR, "px,", wrist_depthR, "m")
    
    # belly
    bellyM = abs(int(shoulderPos[1] + (hipPos[1] - shoulderPos[1]) / 2))
    bellyH = int(max(hipPos[1], hipPos[3]))
    bellyL = int(min(shoulderPos[0],shoulderPos[2],hipPos[0],hipPos[2]))
    bellyR = int(max(shoulderPos[0],shoulderPos[2],hipPos[0],hipPos[2]))
    belly_xy = [shoulder_xyM[0], bellyM] 
    belly_depth = aligned_depth_frame.get_distance(belly_xy[0], belly_xy[1])
    print("INFO: The position of belly is", belly_xy, "px,", belly_depth, "m")
        
    # hip
    for i in range(0, 4): # 0 to 3
        hipPos[i] = int(hipPos[i] // 1)
    hip_xyL = [hipPos[2], hipPos[3]] 
    hip_xyR = [hipPos[0], hipPos[1]] 
    hip_depthL = aligned_depth_frame.get_distance(int(hipPos[0]), int(hipPos[1]))
    hip_depthR = aligned_depth_frame.get_distance(int(hipPos[2]), int(hipPos[3]))
    hipxyzL = rs.rs2_deproject_pixel_to_point(depth_intrin, hip_xyL, hip_depthL)
    hipxyzR = rs.rs2_deproject_pixel_to_point(depth_intrin, hip_xyR, hip_depthR)
    print("INFO: The position of left hip is", hip_xyL, "px,", hip_depthL, "m")
    print("INFO: The position of right hip is", hip_xyR, "px,", hip_depthR, "m")
    
    # get bodyData
    global bodyData, pose_img, selectedcloth_img, pose_keypoints
    
    bodyData = [0,0,0,0]
    shoulderWidth = 0
    chestWidth = 0
    clothingLength = 0
    
    # 0-shoulderWidth
    shoulderWidth = ((shoulderxyzL[0]-shoulderxyzR[0]) ** 2 
            + (shoulderxyzL[1]-shoulderxyzR[1]) ** 2 
            + (shoulderxyzL[2]-shoulderxyzR[2]) ** 2) ** 0.5
    shoulderWidth = shoulderWidth * 100 + 4
    print("INFO: The shoulderWidth is", shoulderWidth, "cm")
    bodyData[0] = shoulderWidth
    
    # 1-chestWidth
    distY = abs(int((hipPos[1] - shoulderPos[1]) / 2))
    # up to down
    for i in range(int(shoulderPos[1]), int(shoulderPos[1]) + distY):
        depthL = aligned_depth_frame.get_distance(int(shoulderPos[0]-20), int(i))
        # print(depthL)
        if (depthL == 0 or depthL > 2.6):
            # left to right
            for j in range(int(shoulderPos[0]-20), int(shoulderPos[0])):
                depthL = aligned_depth_frame.get_distance(int(j), int(i))
                # print(j, " ", i)
                # print(depthL)
                if (depthL > 0 and depthL < 2.6):
                    xyL = [j, i]
                    chestxyzL = rs.rs2_deproject_pixel_to_point(depth_intrin, xyL, depthL)
                    break
            break
        else:
            xyL = [int(shoulderPos[0]-20), i]
            chestxyzL = rs.rs2_deproject_pixel_to_point(depth_intrin, xyL, depthL)
    
    distY = abs(int((hipPos[3] - shoulderPos[3]) / 2))
    # up to down
    for i in range(int(shoulderPos[3]), int(shoulderPos[3]) + distY):
        depthR = aligned_depth_frame.get_distance(int(shoulderPos[2]+20), int(i))
        # print(depthR)
        if (depthR == 0 or depthR > 2.6):
            # left to right
            for j in range(int(shoulderPos[2]+20), int(shoulderPos[2]), -1):
                depthR = aligned_depth_frame.get_distance(int(j), int(i))
                # print(j, " ", i)
                # print(depthR)
                if (depthR > 0 and depthR < 2.6):
                    xyR = [j, i]
                    chestxyzR = rs.rs2_deproject_pixel_to_point(depth_intrin, xyR, depthR)
                    break
            break
        else:
            xyR = [int(shoulderPos[2]+20), i]
            chestxyzR = rs.rs2_deproject_pixel_to_point(depth_intrin, xyR, depthR)
            
    chestWidth = ((chestxyzL[0]-chestxyzR[0]) ** 2 
            + (chestxyzL[1]-chestxyzR[1]) ** 2 
            + (chestxyzL[2]-chestxyzR[2]) ** 2) ** 0.5
    chestWidth = chestWidth * 100 + 3
    print("INFO: The chestWidth is", chestWidth, "cm")
    bodyData[1] = chestWidth
        
    # 2-clothingLength
    clothingLength = (((shoulderxyzL[0]-hipxyzL[0]) ** 2 
            + (shoulderxyzL[1]-hipxyzL[1]) ** 2 
            + (shoulderxyzL[2]-hipxyzL[2]) ** 2) ** 0.5
            + ((shoulderxyzR[0]-hipxyzR[0]) ** 2 
            + (shoulderxyzR[1]-hipxyzR[1]) ** 2 
            + (shoulderxyzR[2]-hipxyzR[2]) ** 2) ** 0.5) / 2
    clothingLength = clothingLength * 100 + 4
    print("INFO: The clothingLength is", clothingLength, "cm")
    bodyData[2] = clothingLength
    print(bodyData)

    json_string = {"version": 1.0, "people": [{"face_keypoints": [],
                                                "pose_keypoints": [
                                                    nose_xy[0], nose_xy[1], nose_depth, 
                                                    shoulder_xyM[0], shoulder_xyM[1], shoulder_depthM,
                                                    shoulder_xyR[0], shoulder_xyR[1], shoulder_depthR,
                                                    elbow_xyR[0], elbow_xyR[1], elbow_depthR,
                                                    wrist_xyR[0], wrist_xyR[1], wrist_depthR, 
                                                    shoulder_xyL[0], shoulder_xyL[1], shoulder_depthL,
                                                    elbow_xyL[0], elbow_xyL[1], elbow_depthL,
                                                    wrist_xyL[0], wrist_xyL[1], wrist_depthL,
                                                    hip_xyR[0], hip_xyR[1], hip_depthR,
                                                    hip_xyL[0], hip_xyL[1], hip_depthL,
                                                    eye_xyR[0], eye_xyR[1], eye_depthR, 
                                                    eye_xyL[0], eye_xyL[1], eye_depthL,
                                                    ear_xyR[0], ear_xyR[1], ear_depthR,
                                                    ear_xyL[0], ear_xyL[1], ear_depthL,],
                                                "hand_right_keypoints": [],
                                                "hand_left_keypoints": []}]} 

    json_keypoints = json.dumps(json_string)
    print(json_keypoints)
    
    # Directly from dictionary
    with open('keypoints.json', 'w') as outfile:
        json.dump(json_keypoints, outfile)
    
    # Using a JSON string
    with open('keypoints.json', 'w') as outfile:
        outfile.write(json_keypoints)
    
    bodyData=[37,42,66]
    pose_img = cv2.imread('020000_0.jpg')
    selectedcloth_img = cv2.imread('020000_1.jpg')
    pose_keypoints = [95.76296296296296 ,85.33333333333333, 1.404750108718872, 
                    99.55555555555556, 130.84444444444443, 1.4445000886917114, 
                    66.84444444444445, 131.31851851851852, 1.4512500762939453, 
                    41.24444444444445 ,180.62222222222223, 1.443000078201294, 
                    10.42962962962963, 221.86666666666667, 1.4037500619888306, 
                    132.26666666666668, 130.37037037037038, 1.4852501153945923, 
                    156.44444444444446, 183.46666666666667, 1.567500114440918, 
                    188.2074074074074, 232.2962962962963, 1.564500093460083, 
                    79.17037037037036, 212.85925925925926, 1.3842500448226929, 
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    116.62222222222222, 214.28148148148148, 1.409250020980835,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 
                    87.70370370370371 ,79.64444444444445, 1.41225004196167, 
                    101.92592592592592, 79.17037037037036, 1.411500096321106,
                    82.96296296296296 ,84.85925925925926, 1.4270000457763672, 
                    109.98518518518519, 82.96296296296296, 1.4280000925064087]
    
    return 0
             
def openLidar(request):
    print(112)
    return StreamingHttpResponse(runLidar(), content_type='multipart/x-mixed-replace; boundary=frame')
'''
def user_showLidar(request):
    # bodyData = runLidar()
    return render(request,'user_showLidar.html',locals())

# def user_pose():
#     ret, pose_frame = cv2.imencode('.jpeg', pose_img)
#     yield (b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n' + pose_frame.tobytes() + b'\r\n')

# def user_pose_img(request):
#     return StreamingHttpResponse(user_pose(), content_type='multipart/x-mixed-replace; boundary=frame')

# def user_selectedcloth():
#     ret, selectedcloth_frame = cv2.imencode('.jpeg', selectedcloth_img)
#     yield (b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n' + selectedcloth_frame.tobytes() + b'\r\n')

# def user_selectedcloth_img(request):
#     return StreamingHttpResponse(user_selectedcloth(), content_type='multipart/x-mixed-replace; boundary=frame')
    
def user_selectCloth(request):
    cloths = Cloth.objects.all()
    context = {
        'app': cloths,
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
    
def shop_manual(request):
    return render(request,'shop_manual.html',{})


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

def getEdgeAndLebel(clothImage,humanImage):
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
    

def generateImage(labelImg,poseImg,colorImg,colorMaskImg,edgeImg,maskImg,keypoints):
    
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
        Image.fromarray(rgb)
        imgByteArr = io.BytesIO()
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
    
'''
def user_showResult(request):
    bodyDataName = ["肩寬","胸寬","身長"]
    size_str = ""
    size_cnt = []
    size_result = ""
    bodyData=[37,42,66]
    # size chart, need to import from database
    chart = [[35, 40, 42, 43, 46],
            [49, 53, 57, 58, 62],
            [70, 75, 78, 81, 82],
            [30, 32, 33, 34, 35]]

    # compare with size chart
    for i in range(0, 3):
        bodyData[i] = np.round(bodyData[i],2)
        bodyData[i] = float(bodyData[i])
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
    #get user selection of cloth image and data
    
    #test
    edgeImg,labelImg=getEdgeAndLebel(selectedcloth_img, pose_img)
    maskImg=Image.open('00000.png').convert('L')
    colorMaskImg=Image.open('00000_test.png').convert('L')
    resultImage_uri=generateImage(labelImg, pose_img, selectedcloth_img, colorMaskImg, edgeImg, maskImg, pose_keypoints)
    
    """
    cloth = NULL
    cloth_data=NULL
    if request.method == "POST":
        print(request.POST['cloth'])
        cloth=Cloth.objects.get(id=request.POST['cloth'])
        cloth_data=Cloth_data.objects.get(image_ID=request.POST['cloth'])
        print(cloth_data)
    """
    context = {
        'bodyDataList': bodyDataList,
        'pose_keypoints': pose_keypoints,
        'pose_img': pose_img,
        'selectedcloth_img': selectedcloth_img,
        'size_result': size_result,
        'resultImage':resultImage_uri,
    }
    
    return render(request,'user_showResult.html', context)
'''

    
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
    p_keypoints=[95.76296296296296 ,85.33333333333333, 1.404750108718872, 
                    99.55555555555556, 130.84444444444443, 1.4445000886917114, 
                    66.84444444444445, 131.31851851851852, 1.4512500762939453, 
                    41.24444444444445 ,180.62222222222223, 1.443000078201294, 
                    10.42962962962963, 221.86666666666667, 1.4037500619888306, 
                    132.26666666666668, 130.37037037037038, 1.4852501153945923, 
                    156.44444444444446, 183.46666666666667, 1.567500114440918, 
                    188.2074074074074, 232.2962962962963, 1.564500093460083, 
                    79.17037037037036, 212.85925925925926, 1.3842500448226929, 
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    116.62222222222222, 214.28148148148148, 1.409250020980835,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 
                    87.70370370370371 ,79.64444444444445, 1.41225004196167, 
                    101.92592592592592, 79.17037037037036, 1.411500096321106,
                    82.96296296296296 ,84.85925925925926, 1.4270000457763672, 
                    109.98518518518519, 82.96296296296296, 1.4280000925064087]

    if request.method == "POST":
        form = ClothesDataModelForm(request.POST)
        #print(request.POST['image_ID'])
        if form.is_valid():
            cloth_info=form.cleaned_data
            cloth_info['image_ID']=cloths.id
            print("cloth_info:",cloth_info)
            Cloth_data.objects.create(**cloth_info)
            img_gray,parsing_result=getEdgeAndLebel(cloth_img,model_img)
            resultImage=generateImage(parsing_result, model_img, cloth_img, colorMaskImg, img_gray, maskImg, p_keypoints,)
            print("result_img:",resultImage)

    context = {
        'app': cloths,
        'text': form,
        'resultImage':resultImage
    }
    context['text'].fields['image_ID'].initial=cloths.id    
    return render(request,'cloth_preview.html',context)