import pandas as pd

def modality_mismatch(video_intervals,audio_intervals):
    modality_mismatch = []

    for row in range(len(video_intervals.index)):
        if video_intervals.iloc[row]['emotion'] != audio_intervals.iloc[row]['emotion']:
            modality_mismatch.append({'start':video_intervals.iloc[row]['start'],
                                     'end':video_intervals.iloc[row]['end'],
                                     'video_emotion':video_intervals.iloc[row]['emotion'],
                                     'audio_emotion':audio_intervals.iloc[row]['emotion']})
    modality_mismatch = pd.DataFrame(modality_mismatch)
    modality_mismatch.to_csv('statistics/modality_mismatch.csv',index=False)
    return modality_mismatch


def modality_duration(video_intervals,audio_intervals):
    modality_duration = []

    for row in range(len(video_intervals.index)):
        modality_duration.append({'duration':(video_intervals.iloc[row]['end']-video_intervals.iloc[row]['start']),
                                  'start':video_intervals.iloc[row]['start'],
                                  'end':video_intervals.iloc[row]['end'],
                                  'video_emotion':video_intervals.iloc[row]['emotion'],
                                  'audio_emotion':audio_intervals.iloc[row]['emotion']})
    modality_duration = pd.DataFrame(modality_duration)
    modality_duration.to_csv('statistics/modality_duration.csv',index=False)
    return modality_duration
    

#video_intervals = pd.read_csv('statistics/video_intervals.csv')

#audio_intervals = pd.read_csv('statistics/audio_intervals.csv')


#modality_mismatch = modality_mismatch(video_intervals, audio_intervals)
#print(modality_mismatch)

#modality_duration = modality_duration(video_intervals,audio_intervals)
#print(modality_duration)


