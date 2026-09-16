import torch
import io
import soundfile as sf
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

MODEL_NAME = "openai/whisper-tiny"
# Используем имеющиеся конфигурации FLEURS
LANGUAGES = ["en_us", "de_de", "nl_nl"]

print("1. Инициализация процессора и модели...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Отключаем дефолтные предупреждения генерации
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, peft_config)

print("2. Подготовка датасета FLEURS...")
train_list = []
eval_list = []

for lang in LANGUAGES:
    print(f"Загрузка конфигурации: {lang}...")
    ds = load_dataset("google/fleurs", lang)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    # Берем по 60 сэмплов для тренировки и по 15 для валидации
    train_list.append(ds["train"].select(range(60)))
    eval_list.append(ds["validation"].select(range(15)))

raw_train = concatenate_datasets(train_list).shuffle(seed=42)
raw_eval = concatenate_datasets(eval_list)

def prepare_dataset(batch):
    audio_bytes = batch["audio"]["bytes"]
    array, sr = sf.read(io.BytesIO(audio_bytes))
    
    batch["input_features"] = processor.feature_extractor(array, sampling_rate=sr).input_features[0]
    text = batch["transcription"] if batch["transcription"] else batch["raw_transcription"]
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

print("Предобработка данных...")
train_data = raw_train.map(prepare_dataset, remove_columns=raw_train.column_names)
eval_data = raw_eval.map(prepare_dataset, remove_columns=raw_eval.column_names)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.input_ids == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-accent-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=10,
    max_steps=60,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=20,
    eval_steps=20,
    logging_steps=10,
    report_to=["none"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("\nЗапуск обучения...")
trainer.train()

model.save_pretrained("./whisper-accent-lora-final")
processor.save_pretrained("./whisper-accent-lora-final")
print("\nОбучение завершено! Модель сохранена в ./whisper-accent-lora-final")