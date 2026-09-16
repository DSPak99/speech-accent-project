import io
import soundfile as sf
from datasets import Audio, load_dataset

print("Загружаем выборку FLEURS...")

# 1. Загружаем датасет
dataset = load_dataset("google/fleurs", "en_us", split="validation[:5]")

# 2. Отключаем автоматический автодекодер HuggingFace (обходит torchcodec/ffmpeg)
dataset = dataset.cast_column("audio", Audio(decode=False))

print("\n--- СТРУКТУРА ДАТАСЕТА ---")
print(dataset)

# 3. Декодируем первый элемент вручную через soundfile
sample = dataset[0]
audio_bytes = sample["audio"]["bytes"]

# Считываем массив и sampling rate из байтов
array, sampling_rate = sf.read(io.BytesIO(audio_bytes))

text = (
    sample["transcription"]
    if sample["transcription"]
    else sample["raw_transcription"]
)

print("\n--- ПРИМЕР ЭЛЕМЕНТА ---")
print(f"Текст (transcription): {text}")
print(f"Аудио sampling rate:  {sampling_rate} Hz")
print(f"Длина аудиопотока:     {len(array)} семплов\n")