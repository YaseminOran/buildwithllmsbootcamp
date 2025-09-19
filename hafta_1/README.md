# Hafta 1: LLM Temelleri ve Python ile NLP'ye Giriş

Bu modülde Large Language Models (LLM) temellerini ve Hugging Face Transformers kütüphanesiyle NLP'ye giriş yapacaksınız.

## 📋 Dosya İçeriği

### 1. `turkish_simple.py`
- **Amaç**: Türkçe NLP için temel text processing
- **İçerik**: Text temizleme, tokenization, basit analizler
- **Öğrenilecekler**: Python ile text işleme temel kavramları

### 2. `microsoft.py` - DialoGPT Konuşma Modeli
- **Model**: Microsoft DialoGPT-medium
- **Amaç**: Conversation AI ve text generation
- **ÖNEMLI**: Bu model çok tutarlı değildir, bu normaldir!

#### 🤖 DialoGPT Gerçek Davranışı:
```
Beklenen: "Language models are AI systems..."
Gerçek: "But what does it say? Human: The model says human."

Beklenen: "AI is the simulation of human intelligence..."  
Gerçek: "I'm not human."

Beklenen: "Machine learning works by..."
Gerçek: "How does machine learning work?"
```

#### 📚 Neden Böyle Cevaplar Veriyor?
- **Reddit/Forum Data**: Casual konuşma verisiyle eğitilmiş
- **Conversation-focused**: Bilgi vermekten çok sohbet etmeye odaklı
- **Limited Knowledge**: Spesifik teknik konularda eksik
- **Inconsistent**: Bazen yardımcı, bazen değil
- **Bu tamamen normal**: Production'da ChatGPT gibi instruction-tuned modeller kullanılır

#### 💡 Bu Ne Öğretiyor?
- **Model Limitations**: Her AI modeli her görevi iyi yapmaz
- **Data Matters**: Model ancak eğitildiği veri kadar iyidir
- **Purpose-built Models**: Farklı görevler için farklı modeller gerekir
- **Real Expectations**: AI'ın gerçek sınırlarını anlama

### 3. `qwen.py` - Qwen 2.5 Text Generation
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Amaç**: Modern instruction-following model deneyimi
- **İyileştirmeler**: Text generation'dan sadece bot cevabını çıkarma

#### 🔧 Teknik Çözüm:
```python
# Problem: Pipeline tüm metni döndürür (prompt + response)
generated_text = response[0]["generated_text"]

# Çözüm: Sadece yeni kısmı al
bot_response = generated_text[len(prompt):].strip()
```

## 🔧 Kurulum

### 1. Sanal Ortamı Aktifleştir
```bash
cd hafta_1
source llm_1/bin/activate  # Mac/Linux
# veya
llm_1\Scripts\activate     # Windows
```

### 2. API Token Ayarla
`.env` dosyasında Hugging Face token'ınızı ayarlayın:
```
HF_TOKEN=hf_your_token_here
```

### 3. Çalıştırma
```bash
python turkish_simple.py
python microsoft.py
python qwen.py
```

## 📖 Öğrenme Hedefleri

### Teknik Beceriler
- ✅ Hugging Face Transformers kullanımı
- ✅ Text generation pipeline'ları
- ✅ Model loading ve configuration
- ✅ Token management ve authentication

### AI/ML Kavramları  
- ✅ **Model Types**: Conversation vs Instruction-tuned models
- ✅ **Data Impact**: Training data'nın model davranışına etkisi
- ✅ **Realistic Expectations**: AI'ın gerçek sınırları
- ✅ **Text Generation**: Prompt engineering temel kavramları

## ⚠️ Önemli Notlar

### DialoGPT Beklenmedik Sonuçları
- **Normal Durum**: Saçma cevaplar vermesi beklenen bir durumdur
- **Educational Value**: Gerçek model limitlerini gösterir
- **Production Reality**: Gerçek uygulamalarda daha stabil modeller kullanılır

### Model Performance
- **Hardware**: GPU varsa daha hızlı çalışır
- **Memory**: Büyük modeller daha fazla RAM tüketir
- **Network**: Model indirme ilk seferde zaman alabilir

### Token Limits
- **Hugging Face**: Ücretsiz token limitleri var
- **Rate Limiting**: Çok hızlı istek göndermeyin
- **Model Access**: Bazı modeller token gerektirir

## 🎯 Sonraki Adım

Hafta 2'de öğreneceğiniz:
- OpenAI API ile prompt engineering
- Function calling ile akıllı chatbot'lar
- Production-ready conversation systems

Bu hafta model sınırlarını görmeniz, gelecek haftalarda daha gelişmiş çözümleri anlayabilmeniz için değerlidir.

---

**💡 Hatırlatma**: DialoGPT'nin garip cevapları sizi şaşırtmasın! Bu, AI'ın gerçek doğasını anlamanız için önemli bir deneyim. 🤖