# video_and_speech_emotion_recognition
В данном проекте реализовано распознавание эмоций по видеоряду и аудиоряду. Распознавание по аудиоряду производится для русского языка. В данный момент распознавание проводится только для одного человека. На основании распознавания сопоставляются данные по обеим модальностям и строится статистика.

## Описание проекта
### Алгоритм
Программа принимает на вход видео, содержащее выступление одного человека. Для каждого кадра видео проводится обнаружение лица и анализ эмоции в случае, если лицо было обнаружено. Выделяются временные интервалы, во время которых выступающий испытывал одну и ту же эмоцию. Данные заносятся в соответстующую таблицу.
Затем для каждого из обнаруженных интервалов проводится анализ аудиодорожки, выделяются эмоциональная окраска голоса.
На основании составленных таблиц проводится анализ полученной информации. Составляется таблица, содержащая интервалы, для которых не соответствуют эмоции. Также составляется таблица, содержащая значения длительности интервалов, их граничных значений и предсказанных эмоций по обеим модальностям. Вычисляется количество переходов между граничными состояниями.

### Входные данные
- Видео выступления
### Выходные данные
- Таблица, содержащая временные интервалы и предсказанную соответствующую эмоцию по видео
- Таблица, содержащая временные интервалы и предсказанную соответствующую эмоцию по аудио
- Таблица несоответствия эмоций по обеим модальностям для одного интервала
- Таблица длительности интервалов, граничных значений, а также предсказанных по обеим модальностям эмоций
Все выходные данные сохраняются в папку statistics

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

### Альтернативный способ
Тут будет описание под докер

### Демо
Демонстрацию работы можно увидеть [здесь](https://github.com/RuslanAntonov/video_and_speech_emotion_recognition/blob/main/video_and_speech_emotion_recognition.ipynb)

## Дальнейшее развитие
*Выделено несколько способов улучшить работу системы:*
В работе использовались предобученные системы. Есть предположение, что после обучения данных моделей на иных наборах данных, будет возможно получить лучший результат.
- Сеть, проводящая предсказание по видео, обучена на датасете, состоящем из фотографий, где эмоции ярко выражены. Встретившись с эмоциями, проявляемыми в повседневной жизни, сеть бывает не способна адекватно предсказать их эмоции, особенно это по какой-то причине касается изображений женщин.
- Модель распознавания речи требует дополнительного тестирования на реальных данных. Также для нее возможно ввести определение значения интенсивности предсказываемой эмоции, что не было реализованно в данной версии программы. 
Также для расширения функционала возможно применить распознавание эмоций по третьей модальности - тексту. Для этого необходимо необходимо применить систему, способную распознавать речь на заданном интервале, преобразовывать ее в текст, а затем проводить предсказание с помощью еще одной сети. В качестве промежуточного варианта возможно использовать заранее заготовленную текстовую расшифровку выступления, но для этого необходимо решить задачу, как сопоставлять текст расшифровки с временными интервалами.

## Источники
- [Модель для визуального распознавания](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)
- [Модель для распознавания голоса](https://huggingface.co/Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition)
- [Видео](https://youtu.be/ycHYHOGmKLY)
