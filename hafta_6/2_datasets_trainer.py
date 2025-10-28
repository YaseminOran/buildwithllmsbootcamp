"""
Datasets + Trainer Kullanımı

Bu script, Hugging Face Datasets ve Trainer kullanarak 
model fine-tuning sürecini detaylı bir şekilde gösterir.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_and_prepare_dataset():
    """
    Dataset yükleme ve hazırlama
    """
    print("Dataset yükleniyor...")
    
    # Örnek sentiment analysis dataset'i
    # Gerçek projede load_dataset("imdb") gibi kullanabilirsiniz
    sample_data = {
        "text": [
            "Bu film gerçekten harikaydı! Çok beğendim.",
            "Berbat bir film, hiç beğenmedim.",
            "Ortalama bir yapım, fena değil.",
            "Muhteşem oyunculuk ve senaryo!",
            "Sıkıcı ve anlamsız bir film.",
            "Güzel bir aile filmi, tavsiye ederim.",
            "Vakit kaybı, izlemeyin.",
            "Etkileyici görsel efektler ve müzik.",
            "Hayal kırıklığı yaratan bir devam filmi.",
            "Mükemmel yönetim ve karakterler."
        ],
        "label": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 1: pozitif, 0: negatif
    }
    
    # Dataset oluştur
    dataset = Dataset.from_dict(sample_data)
    
    # Train/validation split
    train_dataset = dataset.select(range(8))
    eval_dataset = dataset.select(range(8, 10))
    
    return train_dataset, eval_dataset

def setup_tokenizer_and_model(model_name="distilbert-base-uncased"):
    """
    Model ve tokenizer kurulumu
    """
    print(f"Model yükleniyor: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification
    )
    
    return tokenizer, model

def tokenize_dataset(dataset, tokenizer, max_length=128):
    """
    Dataset tokenization
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Dynamic padding kullanacağız
            max_length=max_length
        )
    
    # Batch tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    """
    Evaluation metrics hesaplama
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def setup_training_arguments():
    """
    Training arguments konfigürasyonu
    """
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # WandB/TensorBoard kapalı
        seed=42,
    )
    
    return training_args

def train_model_with_trainer():
    """
    Trainer kullanarak model eğitimi
    """
    print("=== Model Eğitimi Başlıyor ===")
    
    # 1. Dataset hazırlığı
    train_dataset, eval_dataset = load_and_prepare_dataset()
    print(f"Train dataset boyutu: {len(train_dataset)}")
    print(f"Eval dataset boyutu: {len(eval_dataset)}")
    
    # 2. Model ve tokenizer
    tokenizer, model = setup_tokenizer_and_model()
    
    # 3. Tokenization
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    eval_tokenized = tokenize_dataset(eval_dataset, tokenizer)
    
    # 4. Data collator (dynamic padding için)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 5. Training arguments
    training_args = setup_training_arguments()
    
    # 6. Trainer oluşturma
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 7. Eğitim
    print("Eğitim başlıyor...")
    trainer.train()
    
    # 8. Evaluation
    print("Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Evaluation sonuçları: {eval_results}")
    
    # 9. Model kaydetme
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    print("Model başarıyla kaydedildi!")
    
    return trainer, model, tokenizer

def demonstrate_dataset_operations():
    """
    Dataset operasyonlarını gösterir
    """
    print("\n=== Dataset Operasyonları ===")
    
    # Dataset oluşturma
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    print("Dataset özellikleri:")
    print(f"Columns: {train_dataset.column_names}")
    print(f"Features: {train_dataset.features}")
    print(f"İlk örnek: {train_dataset[0]}")
    
    # Dataset filtering
    positive_samples = train_dataset.filter(lambda x: x["label"] == 1)
    print(f"Pozitif örnekler: {len(positive_samples)}")
    
    # Dataset mapping
    def add_length(example):
        example["text_length"] = len(example["text"])
        return example
    
    dataset_with_length = train_dataset.map(add_length)
    print(f"Uzunluk eklenmiş örnek: {dataset_with_length[0]}")
    
    # Batch processing
    def uppercase_text(batch):
        batch["text"] = [text.upper() for text in batch["text"]]
        return batch
    
    uppercase_dataset = train_dataset.map(uppercase_text, batched=True)
    print(f"Uppercase örnek: {uppercase_dataset[0]['text']}")

def demonstrate_training_strategies():
    """
    Farklı eğitim stratejilerini gösterir
    """
    print("\n=== Eğitim Stratejileri ===")
    
    strategies = {
        "Epoch-based": "Her epoch sonunda değerlendirme",
        "Steps-based": "Belirli step sayısında değerlendirme", 
        "No evaluation": "Sadece eğitim, değerlendirme yok",
        "Early stopping": "Performans düşünce erken durdurma"
    }
    
    for strategy, description in strategies.items():
        print(f"- {strategy}: {description}")
    
    print("\nTrainingArguments önemli parametreleri:")
    print("- learning_rate: Öğrenme oranı (varsayılan: 5e-5)")
    print("- weight_decay: L2 regularization (varsayılan: 0.0)")
    print("- warmup_steps: Learning rate warm-up")
    print("- gradient_accumulation_steps: Batch boyutunu artırma")
    print("- fp16: Mixed precision training")

if __name__ == "__main__":
    print("Datasets + Trainer Demonstrasyonu")
    print("=" * 40)
    
    # Dataset operasyonlarını göster
    demonstrate_dataset_operations()
    
    # Training stratejilerini göster
    demonstrate_training_strategies()
    
    # Model eğitimi (uncomment to run)
    # trainer, model, tokenizer = train_model_with_trainer()