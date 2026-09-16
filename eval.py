import torch
import io
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import evaluate

MODEL_NAME = "openai/whisper-tiny"
LORA_PATH = "./whisper-fleurs-lora-final"

print("1. Загрузка процессора и моделей...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")

# Базовая модель
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Модель с LoRA
lora_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)

# Переводим в режим инференса
base_model.eval()
lora_model.eval()

print("2. Загрузка тестового аудио из FLEURS...")
dataset = load_dataset("google/fleurs", "en_us", split="test[:10]")
dataset = dataset.cast_column("audio", Audio(decode=False))

wer_metric = evaluate.load("wer")

base_preds, lora_preds, references = [], [], []

print("\n--- СРАВНЕНИЕ РЕЗУЛЬТАТОВ ---")
for i, sample in enumerate(dataset):
    # Декодируем аудио
    array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
    inputs = processor(array, sampling_rate=sr, return_tensors="pt")
    
    reference = sample["transcription"] if sample["transcription"] else sample["raw_transcription"]
    references.append(reference)

    # Генерация предсказаний
    with torch.no_grad():
        base_tokens = base_model.generate(inputs.input_features)
        lora_tokens = lora_model.generate(inputs.input_features)

    base_text = processor.batch_decode(base_tokens, skip_special_tokens=True)[0]
    lora_text = processor.batch_decode(lora_tokens, skip_special_tokens=True)[0]

    base_preds.append(base_text)
    lora_preds.append(lora_text)

    if i < 3: # Выводим первые 3 примера
        print(f"\n[Пример {i+1}]")
        print(f"Эталон:   {reference}")
        print(f"Base:     {base_text}")
        print(f"LoRA:     {lora_text}")

# Расчет WER
base_wer = wer_metric.compute(predictions=base_preds, references=references)
lora_wer = wer_metric.compute(predictions=lora_preds, references=references)

print("\n" + "="*40)
print(f"Итоговый WER Base Model: {base_wer * 100:.2f}%")
print(f"Итоговый WER LoRA Model: {lora_wer * 100:.2f}%")
print("="*40)