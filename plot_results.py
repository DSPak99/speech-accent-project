import torch
import io
import soundfile as sf
import matplotlib.pyplot as plt
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import evaluate

MODEL_NAME = "openai/whisper-tiny"
LORA_PATH = "./whisper-accent-lora-final"
LANGUAGES = ["en_us", "de_de", "nl_nl"]

# Маппинг конфигураций FLEURS в языковые коды Whisper
LANG_MAP = {
    "en_us": "english",
    "de_de": "german",
    "nl_nl": "dutch"
}

print("1. Loading models and processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
lora_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)

base_model.eval()
lora_model.eval()

wer_metric = evaluate.load("wer")

results_base = {}
results_lora = {}

print("2. Calculating WER per language subset...")

for lang in LANGUAGES:
    print(f"Evaluating subset: {lang}...")
    ds = load_dataset("google/fleurs", lang, split="validation[:20]")
    ds = ds.cast_column("audio", Audio(decode=False))

    base_preds, lora_preds, references = [], [], []
    target_lang = LANG_MAP[lang]

    for sample in ds:
        array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
        inputs = processor(array, sampling_rate=sr, return_tensors="pt")
        
        ref = sample["transcription"] if sample["transcription"] else sample["raw_transcription"]
        references.append(ref)

        # Передаем правильный language token для Whisper decoder
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=target_lang, task="transcribe")

        with torch.no_grad():
            base_tokens = base_model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)
            lora_tokens = lora_model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)

        base_preds.append(processor.batch_decode(base_tokens, skip_special_tokens=True)[0])
        lora_preds.append(processor.batch_decode(lora_tokens, skip_special_tokens=True)[0])

    wer_b = wer_metric.compute(predictions=base_preds, references=references) * 100
    wer_l = wer_metric.compute(predictions=lora_preds, references=references) * 100

    results_base[lang] = round(wer_b, 2)
    results_lora[lang] = round(wer_l, 2)

# 3. Visualizer
print("\n3. Generating plot...")
x = list(LANGUAGES)
width = 0.35
x_indexes = range(len(x))

plt.figure(figsize=(9, 5))
plt.bar([i - width/2 for i in x_indexes], [results_base[l] for l in x], width=width, label='Base Whisper-Tiny', color='#708090')
plt.bar([i + width/2 for i in x_indexes], [results_lora[l] for l in x], width=width, label='LoRA Fine-Tuned', color='#2E8B57')

plt.xlabel('FLEURS Language/Accent Groups', fontsize=11, fontweight='bold')
plt.ylabel('Word Error Rate (WER %)', fontsize=11, fontweight='bold')
plt.title('Performance Comparison: Base vs LoRA Fine-Tuned Whisper', fontsize=13, fontweight='bold')
plt.xticks(ticks=list(x_indexes), labels=x)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i in x_indexes:
    lang = x[i]
    plt.text(i - width/2, results_base[lang] + 1, f"{results_base[lang]}%", ha='center', fontsize=9)
    plt.text(i + width/2, results_lora[lang] + 1, f"{results_lora[lang]}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("wer_comparison_chart.png", dpi=300)
print("Updated chart saved to 'wer_comparison_chart.png'!")