from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

import os

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

import pandas as pd



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def cropp_face(raw_img, faces):
    for (x, y, w, h) in faces:
        x,y,w,h=int(x),int(y),int(w),int(h)
        raw_img = raw_img[y:y + h, x:x + w]
    return raw_img


def outputs_avg_proc(raw_img,transform_test,net):
    #подготовка изображения для inference
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)
    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = Variable(inputs, volatile=True)

    #подготовка inference
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)
    return outputs_avg


def video_emo_rec(video_name,device):
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)

    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'),map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    cut_size = 44

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    interval_start, interval_end = 0.0, 0.0
    interval_exp = ''
    intervals = []

    video = mmcv.VideoReader(video_name)
    fps = round(video.fps)
    print('FPS:', fps)

    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    
    print('Start video processing')
    for i, frame in enumerate(frames):
        #print('Frame num:', i+1)
        
        frame_time = round(i / fps,2)
        #print('Frame time:', frame_time, 'sec')
        
        # Detect faces
        faces, _ = mtcnn.detect(frame)
        frame_crop = np.array(frame)

        if faces is None or len(faces) != 1:
            #print('Cant Predict Expression for Frame\n')
            pass
        else:
            for face in faces:
                raw_img = cropp_face(frame_crop, faces)
                outputs_avg = outputs_avg_proc(raw_img, transform_test, net)
                score = F.softmax(outputs_avg)
                _, predicted = torch.max(outputs_avg.data, 0)
                predicted_exp = str(class_names[int(predicted.cpu().numpy())])
                #print("Frame Expression is %s" %predicted_exp,'\n')
            if interval_exp == predicted_exp:
                pass
            else:
                if i == 0:
                    interval_exp = predicted_exp
                else:
                    interval_end = frame_time
                    intervals.append({'emotion':interval_exp,'start':interval_start,'end':interval_end})
                    interval_start = frame_time
                    interval_exp = predicted_exp
    interval_end = frame_time
    intervals.append({'emotion':predicted_exp,'start':interval_start,'end':interval_end})
    video_intervals = pd.DataFrame(intervals)
    #print(video_intervals)
    video_intervals.to_csv('statistics/video_intervals.csv',index=False)
    return video_intervals



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#video_name = 'video5.mp4'

#video_emo_rec(video_name,device)
