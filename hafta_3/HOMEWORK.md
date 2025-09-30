# 📝 Hafta 3 Ödev - Model Performans Analizi

**Teslim Tarihi**: 1 Hafta  
**Toplam Puan**: 100 puan  
**Format**: Python kodu + Kısa rapor

## 🎯 Ödev Amacı

Bu ödevde, 3 farklı model türünü (GPT, BERT, T5) aynı metin üzerinde test ederek performanslarını karşılaştıracak bir analiz programı geliştireceksiniz.

## 📋 Görev: "LLM Model Performance Analyzer"

### Yapmanız Gereken:

3 farklı model ile aynı metinleri test edip performanslarını karşılaştıran **tek bir program** yazın:

- **GPT-2**: Text generation  
- **BERT**: Sentiment analysis
- **T5**: Text summarization

Her model için ölçmeniz gerekenler:
- ⏱️ Çalışma süresi
- 💾 Memory kullanımı  
- 📊 Çıktı kalitesi

## 🔧 Beklenen Çıktı Formatı:

```
🤖 Model Performance Analyzer
========================================
Test Text: "I love programming with Python. It's amazing for AI!"

📊 GPT-2 Results:
- Task: Text Generation
- Time: 0.45 seconds
- Memory: 2.1 GB
- Output: "I love programming with Python. It's amazing for AI! The language..."

📊 BERT Results:  
- Task: Sentiment Analysis
- Time: 0.23 seconds
- Memory: 1.8 GB
- Output: POSITIVE (confidence: 0.94)

📊 T5 Results:
- Task: Summarization  
- Time: 0.67 seconds
- Memory: 2.3 GB
- Output: "Python programming is great for AI development."

🏆 Performance Summary:
- Fastest Model: BERT (0.23s)
- Most Memory Efficient: BERT (1.8GB)
- Best for Generation: GPT-2
- Best for Analysis: BERT
- Best for Summarization: T5

💡 Recommendation:
For quick sentiment analysis, use BERT.
For creative text generation, use GPT-2.
For summarizing long texts, use T5.
```

## 🚀 Başlangıç Kodu

```python
import torch
import time
import psutil
from transformers import pipeline

def measure_performance(model_pipeline, text, task_name):
    """Model performansını ölçer"""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / 1e9
    
    # Model çalıştır
    if task_name == "text_generation":
        result = model_pipeline(text, max_length=50, num_return_sequences=1, do_sample=False)
    elif task_name == "sentiment_analysis":
        result = model_pipeline(text)
    elif task_name == "summarization":
        result = model_pipeline(text, max_length=30, min_length=10, do_sample=False)
    else:
        result = model_pipeline(text)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used / 1e9
    
    return {
        'task': task_name,
        'time': round(end_time - start_time, 3),
        'memory': round(end_memory, 2), 
        'result': result
    }

def main():
    # Test metni
    test_text = "I love programming with Python. It's amazing for AI development and machine learning projects!"
    
    print("🤖 Model Performance Analyzer")
    print("=" * 40)
    print(f"Test Text: {test_text}\n")
    
    # TODO: Buradan itibaren siz tamamlayacaksınız!
    
    # 1. GPT-2 Pipeline oluşturun
    print("📦 Loading GPT-2...")
    # gpt2_pipeline = pipeline("text-generation", model="gpt2")
    
    # 2. BERT Pipeline oluşturun  
    print("📦 Loading BERT...")
    # bert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # 3. T5 Pipeline oluşturun
    print("📦 Loading T5...")
    # t5_pipeline = pipeline("summarization", model="t5-small")
    
    print("✅ All models loaded!\n")
    
    # 4. Her modeli test edin
    print("🧪 Testing Models...")
    
    # GPT-2 Test
    # gpt2_results = measure_performance(gpt2_pipeline, test_text, "text_generation")
    
    # BERT Test  
    # bert_results = measure_performance(bert_pipeline, test_text, "sentiment_analysis")
    
    # T5 Test
    # t5_results = measure_performance(t5_pipeline, test_text, "summarization")
    
    # 5. Sonuçları yazdırın
    print("\n📊 Results:")
    print("-" * 40)
    
    # GPT-2 Results
    # print(f"📊 GPT-2 Results:")
    # print(f"- Task: Text Generation")
    # print(f"- Time: {gpt2_results['time']} seconds")
    # print(f"- Memory: {gpt2_results['memory']} GB")
    # print(f"- Output: {gpt2_results['result'][0]['generated_text'][:100]}...")
    # print()
    
    # BERT Results
    # print(f"📊 BERT Results:")
    # print(f"- Task: Sentiment Analysis")
    # print(f"- Time: {bert_results['time']} seconds")
    # print(f"- Memory: {bert_results['memory']} GB")
    # print(f"- Output: {bert_results['result'][0]['label']} (confidence: {bert_results['result'][0]['score']:.2f})")
    # print()
    
    # T5 Results
    # print(f"📊 T5 Results:")
    # print(f"- Task: Summarization")
    # print(f"- Time: {t5_results['time']} seconds")
    # print(f"- Memory: {t5_results['memory']} GB")
    # print(f"- Output: {t5_results['result'][0]['summary_text']}")
    # print()
    
    # 6. Performance Summary
    # print("🏆 Performance Summary:")
    # print("-" * 40)
    # Hangi model en hızlı, hangi model en az memory kullanıyor vs.
    
    # 7. Recommendation
    # print("💡 Recommendation:")
    # print("-" * 40)
    # Hangi model ne için uygun, önerileriniz
    
    print("✅ Analiz tamamlandı!")

if __name__ == "__main__":
    main()
```

## 📋 Yapmanız Gerekenler:

### 1. Kodu Tamamlayın (60 puan)
- [ ] GPT-2 pipeline oluşturun
- [ ] BERT pipeline oluşturun
- [ ] T5 pipeline oluşturun
- [ ] Her modeli test edin
- [ ] Sonuçları formatlı yazdırın
- [ ] Performance karşılaştırması yapın

### 2. Analiz Ekleyin (25 puan)
- [ ] Hangi model en hızlı?
- [ ] Hangi model en az memory kullanıyor?
- [ ] Çıktı kalitesi nasıl?
- [ ] Her model ne için uygun?

### 3. Kod Kalitesi (15 puan)
- [ ] Kod çalışıyor ve hata vermiyor
- [ ] Yorumlar ve açıklamalar var
- [ ] Temiz ve okunabilir kod

## 💡 İpuçları

### Teknik İpuçları:
```python
# Pipeline oluşturma
pipeline("text-generation", model="gpt2")
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  
pipeline("summarization", model="t5-small")

# Memory ölçümü
psutil.virtual_memory().used / 1e9  # GB cinsinden

# Süre ölçümü
start_time = time.time()
# ... işlem ...
duration = time.time() - start_time
```

### Analiz İpuçları:
- İlk çalıştırma model yükleme içerir, onu hariç tutun
- Aynı metni birkaç kez test edin, tutarlı mı?
- Hangi model hangi görev için optimize edilmiş?

### Yaygın Hatalar:
- Model isimlerini yanlış yazmak
- Memory ölçümünü yanlış yapmak  
- Sonuçları yanlış parse etmek

## 📊 Değerlendirme Kriterleri

| Kriter | Puan | Açıklama |
|--------|------|----------|
| **Kod Çalışıyor** | 40 puan | Tüm modeller yükleniyor ve çalışıyor |
| **Performance Ölçümü** | 20 puan | Süre ve memory doğru ölçülüyor |
| **Analiz ve Karşılaştırma** | 25 puan | Modeller objektif karşılaştırılmış |
| **Kod Kalitesi** | 15 puan | Temiz, yorumlu, okunabilir kod |

## 🎯 Bonus Puanlar (+10)

- [ ] **Görselleştirme**: Matplotlib ile performans grafiği (+5 puan)
- [ ] **Extra Test**: Farklı metinlerle test (+3 puan)  
- [ ] **Error Handling**: Try-catch ile hata yönetimi (+2 puan)

## 📚 Faydalı Kaynaklar

- [Hugging Face Pipeline Documentation](https://huggingface.co/docs/transformers/pipeline_tutorial)
- [psutil Documentation](https://psutil.readthedocs.io/)
- Hafta 3 ders materyalleri

## ❓ Sık Sorulan Sorular

**S: Model indirme çok yavaş?**  
C: İlk seferde modeller indirilir, normal. Küçük modelleri kullanın.

**S: Memory hatası alıyorum?**  
C: Daha küçük modeller deneyin: `distilgpt2`, `distilbert-base-uncased`

**S: Sonuçlar her seferde farklı çıkıyor?**  
C: `do_sample=False` parametresini kullanın, daha tutarlı olur.

**S: Hangi modelleri kullanmalıyım?**  
C: 
- GPT: `gpt2` veya `distilgpt2`
- BERT: `distilbert-base-uncased-finetuned-sst-2-english`
- T5: `t5-small`

## 📝 Teslim Formatı

**Dosya adı**: `isim_soyisim_hafta3.py`

**Email konusu**: "[Hafta 3 Ödev] İsim Soyisim"

**İçerik**:
1. Python dosyası (.py)
2. Kısa rapor (2-3 paragraf):
   - Hangi model ne için uygun?
   - En çok neyi öğrendiniz?
   - Gerçek dünyada nasıl kullanırsınız?

---

**İyi çalışmalar! 🚀**

*Bu ödev, gerçek dünya LLM uygulamalarında model seçimi konusunda deneyim kazandıracak.*