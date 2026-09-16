import evaluate
import numpy as np
import soundfile as sf
from transformers import pipeline

# 1. Загружаем модель Whisper
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# 2. Считываем аудиофайл
speech_data, sampling_rate = sf.read("test_audio.mp3")

# Переводим в моно, если файл в стерео
if len(speech_data.shape) > 1 and speech_data.shape[1] == 2:
    speech_data = np.mean(speech_data, axis=1)

# 3. Запускаем расшифровку с флагом return_timestamps=True
result = pipe(
    {"raw": speech_data, "sampling_rate": sampling_rate},
    return_timestamps=True,
)
predicted_text = result["text"]

# 4. Эталонный текст (замените на то, что вы на самом деле сказали!)
reference_text = "My youngest son seems to have grown fond of you, sir. This time he was a squire. But he tells me he will serve no knight but you. He's an unruly boy as you would have noticed. And he's a good lad. Just needs a stern hand, that's all."

# 5. Считаем метрику WER
wer_metric = evaluate.load("wer")
wer = wer_metric.compute(
    predictions=[predicted_text], references=[reference_text]
)

print("\n--- РЕЗУЛЬТАТ ТЕСТА ---")
print(f"Модель распознала: {predicted_text}")
print(f"Эталонный текст:  {reference_text}")
print(f"Ошибок (WER):      {wer * 100:.2f}%\n")