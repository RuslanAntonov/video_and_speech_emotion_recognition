import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

import librosa
import numpy as np

from moviepy.editor import *
from pydub import AudioSegment

import pandas as pd



def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def audio_predict(path, sampling_rate,class_names,feature_extractor,device,model_):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model_(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    predicted = list(scores).index(np.max(scores))
    #print('The Interval Expression is', class_names[predicted])
    return predicted


def get_audio(video_name):
    sound = VideoFileClip(video_name,audio_fps=16000)
    sound.audio.write_audiofile(r"sound.wav")

    sound = AudioSegment.from_wav("sound.wav")
    sound = sound.set_channels(1)
    sound.export("sound.wav", format="wav")

def audio_emo_rec(video_name,device,video_intervals):
    config = AutoConfig.from_pretrained('Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition', trust_remote_code=True)
    model_ = AutoModel.from_pretrained("Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition", trust_remote_code=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_.to(device)

    
    class_names = ['Angry', 'Disgust', 'Surprise', 'Fear', 'Happy', 'Neutral', 'Sad']

    intervals = []


    print('Start audio processing')


    get_audio(video_name)

    sound = AudioSegment.from_wav("sound.wav")
    

    for row in range(len(video_intervals.index)):
        start, end = video_intervals.iloc[row]['start'], video_intervals.iloc[row]['end']
        extract = sound[start*1000:end*1000]
        extract.export("extract.wav", format="wav")
        result = audio_predict("extract.wav", 16000, class_names, feature_extractor, device, model_)
        intervals.append({'emotion':class_names[result],'start':start,'end':end})

    audio_intervals = pd.DataFrame(intervals)
    audio_intervals.to_csv('statistics/audio_intervals.csv',index=False)
    #print(audio_intervals)
    return audio_intervals



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#video_name = 'video5.mp4'
#video_intervals = pd.read_csv('video_intervals.csv')
#audio_emo_rec(video_name, device, video_intervals)


