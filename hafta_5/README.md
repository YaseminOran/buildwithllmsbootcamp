# Hafta 5: İleri Düzey LangChain - Chain, Memory, Tools ve Streaming

Bu hafta **LangChain** framework'ünün ileri seviye özelliklerini öğreneceksiniz. Chain yapıları, memory yönetimi, tool kullanımı ve streaming output konularında derinlemesine bilgi edineceksiniz.

## 📋 İçerik

### 🔗 1. Chain Yapıları (`1_chains_basic.py`)
- **LLMChain**: Temel zincir yapısı
- **SimpleSequentialChain**: Basit sıralı zincirler
- **SequentialChain**: Karmaşık sıralı zincirler  
- **Custom Output Parser**: Özel çıktı işleyiciler

**Öğrenecekleriniz:**
- Chain'lerin nasıl birleştirildiği
- Sıralı işlem akışları oluşturma
- Çıktı formatını özelleştirme

### 🧠 2. Memory Kullanımı (`2_memory_examples.py`)
- **ConversationBufferMemory**: Tüm konuşmayı hatırlama
- **ConversationBufferWindowMemory**: Son N mesajı hatırlama
- **ConversationSummaryMemory**: Konuşma özetleme
- **ConversationSummaryBufferMemory**: Hibrit yaklaşım
- **ConversationTokenBufferMemory**: Token limiti ile memory

**Öğrenecekleriniz:**
- Farklı memory türleri ve kullanım alanları
- Memory optimizasyon teknikleri
- Konuşma geçmişi yönetimi

### 🛠️ 3. Tools ve Agents (`3_tools_and_agents.py`)
- **Basit Tool Kullanımı**: Matematik, zaman, metin işleme
- **Özel Tool Sınıfları**: Hava durumu, haber API'leri
- **Memory ile Agents**: Konuşma geçmişi tutan agent'lar
- **ReAct Agents**: Modern agent yapıları

**Öğrenecekleriniz:**
- Tool oluşturma ve kullanma
- Agent türleri ve özellikleri
- Karmaşık görevler için agent tasarımı

### 🎯 4. Senaryo Bazlı Uygulamalar (`4_scenario_applications.py`)
- **Müşteri Hizmetleri Bot'u**: Sipariş takibi, destek sistemi
- **İçerik Oluşturma Asistanı**: Araştırma, planlama, yazım
- **Eğitim Planlama Asistanı**: Kişisel öğrenim planları

**Öğrenecekleriniz:**
- Gerçek hayat senaryoları için LangChain kullanımı
- End-to-end uygulama geliştirme
- İş süreçlerinin otomasyonu

### 🌊 5. Streaming ve Canlı Veri (`5_streaming_examples.py`)
- **Temel Streaming**: Real-time output
- **Özel Callback Handler'lar**: İlerleme gösterimi
- **Real-time Chat Bot**: Canlı sohbet
- **Async Streaming**: Asenkron işlemler

**Öğrenecekleriniz:**
- Streaming output implementasyonu
- Kullanıcı deneyimi iyileştirme
- Real-time uygulamalar geliştirme

## 🚀 Kurulum

### 1. Virtual Environment Oluştur
```bash
# Otomatik kurulum (önerilen)
python setup_venv.py

# Manuel kurulum
python -m venv hafta5_env

# Windows
hafta5_env\Scripts\activate

# Mac/Linux  
source hafta5_env/bin/activate
```

### 2. Paketleri Yükle
```bash
pip install -r requirements.txt
```

### 3. Environment Değişkenlerini Ayarla
`.env` dosyası oluşturun:
```
OPENAI_API_KEY=your-openai-api-key-here
```

### 4. Test Et
```bash
python test_installation.py
```

## 📚 Dosya Açıklamaları

| Dosya | Açıklama |
|-------|----------|
| `1_chains_basic.py` | Chain yapıları ve kullanımı |
| `2_memory_examples.py` | Memory türleri ve optimizasyonu |
| `3_tools_and_agents.py` | Tool oluşturma ve agent kullanımı |
| `4_scenario_applications.py` | Gerçek hayat senaryoları |
| `5_streaming_examples.py` | Streaming ve real-time örnekler |
| `setup_venv.py` | Otomatik kurulum scripti |
| `test_installation.py` | Kurulum test scripti |
| `requirements.txt` | Gerekli paketler |
| `homework.md` | Haftalık ödev |

## 🎓 Çalışma Sırası

1. **Kurulum yapın** ve test edin
2. **Chain yapılarını** öğrenin (`1_chains_basic.py`)
3. **Memory türlerini** keşfedin (`2_memory_examples.py`)
4. **Tool ve Agent'ları** deneyin (`3_tools_and_agents.py`)
5. **Senaryo uygulamalarını** inceleyin (`4_scenario_applications.py`)
6. **Streaming özelliklerini** test edin (`5_streaming_examples.py`)
7. **Homework'u** tamamlayın

## 🔧 Önemli Konseptler

### Chain Türleri
- **LLMChain**: Temel yapı taşı
- **Sequential**: Sıralı işlemler
- **Router**: Koşullu yönlendirme
- **Transform**: Veri dönüştürme

### Memory Stratejileri
- **Buffer**: Tüm geçmiş
- **Window**: Sınırlı geçmiş  
- **Summary**: Özetlenmiş geçmiş
- **Token-based**: Token limiti

### Agent Türleri
- **Zero-shot ReAct**: En basit
- **Conversational**: Memory destekli
- **ReAct**: Modern yapı
- **Custom**: Özel agent'lar

### Streaming Faydaları
- Daha iyi kullanıcı deneyimi
- Real-time feedback
- Progressive loading
- Responsive arayüzler

## 🐛 Sorun Giderme

### API Key Hatası
```
Error: OpenAI API key not found
```
**Çözüm**: `.env` dosyasında API anahtarınızı kontrol edin

### Import Hatası
```
ImportError: No module named 'langchain'
```
**Çözüm**: Virtual environment aktif mi kontrol edin, requirements.txt yükleyin

### Memory Hatası
```
Memory limit exceeded
```
**Çözüm**: Memory türünü değiştirin (WindowMemory veya SummaryMemory kullanın)

### Streaming Hatası  
```
Streaming not supported
```
**Çözüm**: LLM'i `streaming=True` parametresi ile oluşturun

## 📊 Performans İpuçları

### Memory Optimizasyonu
- Uzun konuşmalar için `ConversationSummaryMemory` kullanın
- Token limiti ile `ConversationTokenBufferMemory` tercih edin
- Gereksiz memory temizleyin

### Chain Optimizasyonu
- Paralel işlemler için async kullanın
- Cache mekanizması ekleyin
- Error handling implementasyonu yapın

### Agent Optimizasyonu
- Tool açıklamalarını net yazın
- Max iterations limitini ayarlayın
- Verbose modu ile debug yapın

## 🎯 Ödev Hazırlığı

Ödev için aşağıdaki konuları anladığınızdan emin olun:

1. **Chain Birleştirme**: Farklı chain'leri nasıl birleştirirsiniz?
2. **Memory Seçimi**: Hangi durumlarda hangi memory türünü kullanırsınız?
3. **Tool Oluşturma**: Kendi tool'larınızı nasıl yazarsınız?
4. **Streaming**: Real-time uygulamalar için streaming nasıl kullanılır?

## 📖 Ek Kaynaklar

- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Chain Examples](https://python.langchain.com/docs/modules/chains)
- [Memory Guide](https://python.langchain.com/docs/modules/memory)
- [Agent Cookbook](https://python.langchain.com/docs/modules/agents)

## 💡 İpuçları

- Her örneği adım adım çalıştırın
- Kod üzerinde değişiklikler yaparak deneyin  
- Error mesajlarını okuyun ve anlayın
- Verbose=True ile debug yapın
- API key'inizi güvenli tutun

**Başarılar! 🚀**