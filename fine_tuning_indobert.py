from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch
import gc

""" import transformers
print(transformers.__version__) """

# Membersihkan garbage collector Python
gc.collect()

# Mengosongkan cache CUDA
torch.cuda.empty_cache()

# 1. Dataset Dummy (contoh: analisis sentimen)
data = {
    "text": [
        "Film ini sangat bagus dan menghibur",
        "Saya tidak suka ceritanya terlalu lambat",
        "Aktor utamanya berakting luar biasa",
        "Jalan cerita membosankan dan tidak menarik",
        "Film yang luar biasa, saya sangat menikmatinya",
        "Alur cerita kurang menarik dan datar"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Positif, 0 = Negatif
}

dataset = Dataset.from_dict(data)

# 2. Muat tokenizer
tokenizer = BertTokenizer.from_pretrained("indolem/indobert-base-uncased")

# 3. Tokenisasi teks
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Muat model untuk klasifikasi
model = BertForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=2)

# 5. Definisikan metrik evaluasi
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "report": classification_report(labels, preds)
    }

# 6. Konfigurasi training (dengan CUDA support)
training_args = TrainingArguments(
    output_dir="./results-indobert",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,     # Sesuaikan sesuai VRAM GPU
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(),   # Aktifkan FP16 jika ada GPU
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 7. Buat Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 8. Jalankan fine-tuning
print("🚀 Mulai fine-tuning IndoBERT...")
trainer.train()

# 9. Evaluasi hasil
print("✅ Evaluasi hasil:")
eval_result = trainer.evaluate()
print(f"Akurasi: {eval_result['eval_accuracy']:.2f}")
print(eval_result["eval_report"])

# 10. Simpan model dan tokenizer ke folder tujuan
output_dir = "./model_fine_tune"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n💾 Model dan tokenizer berhasil disimpan di folder: '{output_dir}'")