# Hafta 6: İleri Düzey Model Fine-tuning ve Kişiselleştirme

Bu hafta, derin öğrenme modellerini verimli bir şekilde fine-tune etme ve kişiselleştirme konularını ele alacağız.

## 📚 Konular

### 1. PEFT (Parameter Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation) nedir ve nasıl çalışır?
- QLoRA ile bellek optimizasyonu
- Adapter katmanları
- PEFT ile model boyutunu küçük tutma

### 2. Datasets + Trainer Kullanımı
- Hugging Face Datasets kütüphanesi
- Veri ön işleme ve tokenization
- Trainer sınıfı ile model eğitimi
- TrainingArguments konfigürasyonu

### 3. Inference ve Kişiselleştirilmiş Model
- Fine-tune edilmiş modeli kullanma
- Inference optimizasyonu
- Model deployment stratejileri
- Kişiselleştirilmiş çıktılar üretme

## 🛠 Pratik Uygulamalar

Her konu için hands-on örnekler ve kod snippet'leri içerir.

## 📋 Gereksinimler

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

## 🎯 Öğrenme Hedefleri

Bu hafta sonunda:
- PEFT teknikleri ile verimli fine-tuning yapabileceksiniz
- Datasets ve Trainer kullanarak model eğitimi gerçekleştirebileceksiniz
- Kendi modelinizi inference için kullanabileceksiniz