from video_emo_rec import *
from audio_emo_rec import *
from statistics import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_name = 'video_short.mp4'


#Временные метки интервалов для видео
video_intervals = video_emo_rec(video_name, device)
print(video_intervals)

#Временные метки интервалов для аудио
audio_intervals = audio_emo_rec(video_name,device,video_intervals)
print(audio_intervals)


print('Start statistics')

#Несоответствие в эмоциях для одних и тех интервалов для видео и аудио
modality_mismatch = modality_mismatch(video_intervals, audio_intervals)
print(modality_mismatch)

#Продолжительность интервалов, их начальные и конечные точки, предсказанные значения
modality_duration = modality_duration(video_intervals,audio_intervals)
print(modality_duration)

#Количество переходов между граничными состояниями
print('\nNumber of transitions between boundary states:',
      len(video_intervals.index))
