# video_and_speech_emotion_recognition
В данном проекте реализовано распознавание эмоций по видеоряду и аудиоряду. Распознавание по аудиоряду производится для русского языка. В данный момент распознавание проводится только для одного человека. На основании распознавания сопоставляются данные по обеим модальностям и строится статистика.

## Установка и запуск
Скопируйте этот репозиторий
```
git clone https://github.com/RuslanAntonov/video_and_speech_emotion_recognition.git
cd video_and_speech_emotion_recognition/
```
Установите зависимости
```
pip install -r requirements.txt
```
Запустите main.py
```
python main.py
```

Демонстрацию работы можно увидеть [здесь](https://github.com/RuslanAntonov/video_and_speech_emotion_recognition/blob/main/video_and_speech_emotion_recognition.ipynb)

## Дальнейшее развитие
В процессе написания

## Источники
- [Модель для визуального распознавания](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)
- [Модель для распознавания голоса](https://huggingface.co/Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition)
- [Видео](https://youtu.be/ycHYHOGmKLY)
