# Hafta 3: AutoTokenizer, AutoModel ve Pipeline Optimizasyonu

Bu hafta, Hugging Face Transformers kütüphanesinin temel bileşenlerini derinlemesine inceleyerek model performansını optimize etme yöntemlerini öğreniyoruz.

## 📚 İçerik

### 1. AutoTokenizer & AutoModel Yapısı + Pipeline ile Hızlı Model Çağırma
**Dosya:** `01_autotokenizer_automodel.py`

- AutoTokenizer kullanımı ve tokenization işlemleri
- AutoModel ile manuel model çağırma
- Pipeline ile hızlı model kullanımı
- Manual vs Pipeline performans karşılaştırması
- Özelleştirilmiş pipeline örnekleri

**Ana Konular:**
- Tokenization (encode/decode)
- Model inference
- Pipeline kullanımı (sentiment-analysis, text-generation, qa, fill-mask)
- Feature extraction
- Performans optimizasyonu

### 2. GPT, BERT ve T5 Modellerinin Farkları ve Pipeline Entegrasyonu
**Dosya:** `02_gpt_bert_t5_comparison.py`

- Model mimarilerinin karşılaştırması
- Her modelin güçlü yönleri ve kullanım alanları
- Pipeline ile üç modeli tek satırda test etme
- Model boyutu ve performans karşılaştırması

**Model Özellikleri:**

| Model | Mimari | Güçlü Yönler | Kullanım Alanları |
|-------|--------|--------------|-------------------|
| **GPT** | Decoder-only | Text generation | Creative writing, Conversational AI |
| **BERT** | Encoder-only | Bidirectional understanding | Classification, NER, QA |
| **T5** | Encoder-decoder | Text-to-text format | Translation, Summarization |

### 3. CPU/GPU Performans Yönetimi ve Model Optimizasyonu
**Dosya:** `03_cpu_gpu_optimization.py`

- Cihaz tespiti ve optimal cihaz seçimi
- CPU optimizasyonu (thread ayarları)
- GPU optimizasyonu (memory management)
- Model quantization (8-bit, dynamic)
- Batch processing optimizasyonu
- Memory efficient inference

**Optimizasyon Teknikleri:**
- `torch.no_grad()` kullanımı
- Memory cleanup (`torch.cuda.empty_cache()`)
- Quantization (BitsAndBytesConfig)
- Batch size optimizasyonu
- Device-specific optimizations

### 4. Pipeline ile GPU/CPU Performansını Ölçme ve Kıyaslama
**Dosya:** `04_performance_measurement.py`

- Performans ölçüm araçları (PerformanceMeter sınıfı)
- Farklı task'lar için benchmark testleri
- Batch size performans analizi
- Model karşılaştırma benchmark'ları
- Detaylı performans raporları

**Ölçülen Metrikler:**
- Inference süresi
- Memory kullanımı (CPU/GPU)
- Throughput (texts/second)
- Device utilization
- Model loading time

## 🚀 Kurulum ve Çalıştırma

### 🔧 Otomatik Kurulum

#### macOS / Linux
```bash
cd hafta_3
chmod +x start.sh
./start.sh
```

#### Windows
```cmd
cd hafta_3
.\start.bat
```

### 📝 Manuel Kurulum

#### 1. Sanal Ortam Oluştur
```bash
# macOS/Linux
python3 -m venv llm_bootcamp_env
source llm_bootcamp_env/bin/activate

# Windows
python -m venv llm_bootcamp_env
llm_bootcamp_env\Scripts\activate.bat
```

#### 2. Bağımlılıkları Yükle
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Modülleri Çalıştır
```bash
# AutoTokenizer ve AutoModel örnekleri
python 01_autotokenizer_automodel.py

# Model karşılaştırması
python 02_gpt_bert_t5_comparison.py

# Performans optimizasyonu
python 03_cpu_gpu_optimization.py

# Performans ölçümü
python 04_performance_measurement.py
```

## 📋 Gereksinimler

```bash
pip install transformers torch torchvision torchaudio
pip install psutil matplotlib numpy
pip install bitsandbytes  # Quantization için (opsiyonel)
```

**GPU Desteği için:**
- CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Apple Silicon: Otomatik olarak MPS desteği

## 🎯 Öğrenme Hedefleri

Bu hafta sonunda şunları öğrenmiş olacaksınız:

1. **AutoTokenizer ve AutoModel** kullanarak manual model çağırma
2. **Pipeline'lar** ile hızlı ve kolay model kullanımı
3. **GPT, BERT, T5** model farklarını ve hangisini ne zaman kullanacağınızı
4. **CPU/GPU optimizasyonu** ile performansı artırma
5. **Performans ölçümü** ve benchmark yapma
6. **Model quantization** ile memory kullanımını azaltma
7. **Batch processing** ile throughput artırma

## 💡 En İyi Uygulamalar

### Performans Optimizasyonu
```python
# ✅ İyi
with torch.no_grad():
    outputs = model(**inputs)

# ❌ Kötü  
outputs = model(**inputs)  # Gradient hesaplanır
```

### Device Yönetimi
```python
# ✅ İyi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# ❌ Kötü
model = model.to("cuda")  # CUDA olmayabilir
```

### Memory Yönetimi
```python
# ✅ İyi
del model
torch.cuda.empty_cache()
gc.collect()

# ❌ Kötü
# Memory leak'e sebep olabilir
```

## 📊 Benchmark Sonuçları

Tipik performans karşılaştırması (örnek sistem):

| Model | Device | Inference Time | Memory Usage |
|-------|--------|----------------|--------------|
| DistilBERT | CPU | 0.045s | 1.2 GB |
| DistilBERT | GPU | 0.012s | 2.1 GB |
| BERT-base | CPU | 0.089s | 2.1 GB |
| BERT-base | GPU | 0.021s | 3.2 GB |

## 🔍 Sorun Giderme

### Yaygın Hatalar

**1. CUDA Out of Memory**
```python
# Çözüm: Batch size'ı azaltın
batch_size = 8  # 32 yerine
torch.cuda.empty_cache()
```

**2. Model Loading Hatası**
```python
# Çözüm: Cihaz uyumluluğunu kontrol edin
device = get_optimal_device()
model = model.to(device)
```

**3. Tokenizer Hatası**
```python
# Çözüm: Padding ve truncation ekleyin
inputs = tokenizer(text, return_tensors="pt", 
                   padding=True, truncation=True)
```

## 📚 Ek Kaynaklar

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [T5 Paper](https://arxiv.org/abs/1910.10683)

## 🎯 Pratik Egzersizler

1. **Kendi Modelinizi Test Edin**: Farklı bir model ile performance benchmark yapın
2. **Custom Pipeline**: Kendi task'ınız için özelleştirilmiş pipeline oluşturun
3. **Optimization Challenge**: Bir modelin performansını %50 artırmaya çalışın
4. **Memory Analysis**: Farklı model boyutlarının memory kullanımını analiz edin

---

**Not:** Bu modüller eğitim amaçlıdır. Production kullanımında güvenlik ve error handling eklemeleri yapılmalıdır.