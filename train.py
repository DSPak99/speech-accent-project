import torch
import io
import soundfile as sf
from datasets import load_dataset, Audio
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

# 1. Загрузка процессора и модели
MODEL_NAME = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# 2. Настройка LoRA (PEFT)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Подготовка датасета (FLEURS en_us)
print("Загрузка датасета...")
raw_dataset = load_dataset("google/fleurs", "en_us")
raw_dataset = raw_dataset.cast_column("audio", Audio(decode=False))

def prepare_dataset(batch):
    # Декодируем байты вручную через soundfile
    audio_bytes = batch["audio"]["bytes"]
    array, sr = sf.read(io.BytesIO(audio_bytes))
    
    # Ресемплинг/извлечение фичей через процессор Whisper
    batch["input_features"] = processor.feature_extractor(
        array, sampling_rate=sr
    ).input_features[0]
    
    # Токенизация текста
    text = batch["transcription"] if batch["transcription"] else batch["raw_transcription"]
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

# Для теста берем небольшие срезы train и validation
train_data = raw_dataset["train"].select(range(50)).map(prepare_dataset, remove_columns=raw_dataset["train"].column_names)
eval_data = raw_dataset["validation"].select(range(10)).map(prepare_dataset, remove_columns=raw_dataset["validation"].column_names)

# 4. Data Collator для выравнивания длин в батче
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

# 5. Метрика WER (Word Error Rate)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 6. Конфигурация обучения
# 6. Конфигурация обучения
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-fleurs-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    warmup_steps=5,
    max_steps=20,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",  # <-- Исправлено здесь!
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=10,
    eval_steps=10,
    logging_steps=5,
    report_to=["none"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,  # <-- Заменено tokenizer на processing_class
)

print("\nЗапуск тестового обучения (20 шагов)...")
trainer.train()

# Сохраняем LoRA адаптер
model.save_pretrained("./whisper-fleurs-lora-final")
processor.save_pretrained("./whisper-fleurs-lora-final")
print("\nОбучение завершено! Модель сохранена в ./whisper-fleurs-lora-final")