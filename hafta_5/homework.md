# Hafta 5 Ödev - İleri Düzey LangChain Projesi

## 🎯 Proje: Akıllı Müşteri Destek Sistemi

Bu ödevde, **LangChain** framework'ünü kullanarak kapsamlı bir müşteri destek sistemi geliştireceksiniz. Sistem; chain yapıları, memory yönetimi, custom tool'lar ve streaming özelliklerini içerecek.

## 📋 Proje Gereksinimleri

### 🏗️ Temel Yapı
Aşağıdaki özellikleri içeren bir sistem oluşturun:

1. **Multi-Chain İşlem Akışı**
2. **Akıllı Memory Yönetimi**  
3. **Custom Tool Integration**
4. **Real-time Streaming Response**
5. **Senaryo Bazlı Test Sistemi**

---

## 🔧 Görev 1: Chain Mimarisi (25 puan)

### Gereksinimler:
Aşağıdaki chain'leri oluşturun:

#### 1.1. Müşteri Analiz Chain'i
```python
class CustomerAnalysisChain:
    # Müşteri mesajını analiz eden chain
    # Input: müşteri mesajı
    # Output: kategori (teknik, billing, genel), aciliyet (düşük/orta/yüksek), dil tonu
```

#### 1.2. Yanıt Üretim Chain'i  
```python
class ResponseGenerationChain:
    # Analiz sonucuna göre uygun yanıt üreten chain
    # Input: analiz sonucu + müşteri mesajı  
    # Output: profesyonel müşteri yanıtı
```

#### 1.3. Kalite Kontrol Chain'i
```python
class QualityControlChain:
    # Üretilen yanıtın kalitesini kontrol eden chain
    # Input: üretilen yanıt
    # Output: kalite skoru + iyileştirme önerileri
```

### Beklenen Çıktı:
- 3 ayrı chain sınıfı
- SequentialChain ile birleştirilmiş ana chain
- Her chain için test örnekleri

---

## 🧠 Görev 2: Gelişmiş Memory Sistemi (25 puan)

### Gereksinimler:

#### 2.1. Hibrit Memory Implementasyonu
```python
class HybridMemorySystem:
    def __init__(self):
        # ConversationSummaryBufferMemory ana memory
        # ConversationBufferWindowMemory son 5 mesaj için
        # Custom metadata storage müşteri bilgileri için
```

#### 2.2. Müşteri Profili Memory
```python
class CustomerProfileMemory:
    def __init__(self):
        # Müşteri özelliklerini saklar
        # Tercih edilen iletişim stili
        # Geçmiş sorun kategorileri  
        # Çözüm memnuniyeti
```

### Beklenen Çıktı:
- İki farklı memory sistemi
- Memory optimize etme fonksiyonları
- Memory durumunu raporlayan fonksiyonlar

---

## 🛠️ Görev 3: Custom Tool Development (25 puan)

### Gereksinimler:

#### 3.1. Ticket Management Tool
```python
class TicketManagementTool(BaseTool):
    name = "ticket_manager"
    description = "Destek bileti oluşturur, günceller ve takip eder"
    
    def _run(self, action, ticket_id=None, details=None):
        # CREATE: yeni ticket oluştur
        # UPDATE: mevcut ticket'ı güncelle  
        # STATUS: ticket durumunu kontrol et
        # LIST: müşteri ticket'larını listele
```

#### 3.2. Knowledge Base Tool
```python
class KnowledgeBaseTool(BaseTool):
    name = "knowledge_base"
    description = "Şirket bilgi tabanından ilgili makaleleri bulur"
    
    def _run(self, query, category=None):
        # Sorgu bazlı makale arama
        # Kategori filtreleme
        # Relevance scoring
```

#### 3.3. Customer Database Tool
```python
class CustomerDatabaseTool(BaseTool):
    name = "customer_db"  
    description = "Müşteri bilgilerini sorgular ve günceller"
    
    def _run(self, customer_id, action="get"):
        # GET: müşteri bilgileri
        # UPDATE: bilgi güncelleme
        # HISTORY: geçmiş etkileşimler
```

### Beklenen Çıktı:
- 3 custom tool sınıfı
- Her tool için comprehensive test fonksiyonu
- Error handling ve validation

---

## 🌊 Görev 4: Streaming Interface (15 puan)

### Gereksinimler:

#### 4.1. Real-time Response Handler
```python
class CustomerServiceStreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        # Progressive response gösterimi
        # Typing indicator
        # Response formatting
```

#### 4.2. Interactive Chat Interface
```python
class InteractiveChatSystem:
    def start_session(self, customer_id):
        # Chat session başlat
        # Streaming response ile gerçek zamanlı sohbet
        # Session logging
```

### Beklenen Çıktı:
- Custom streaming handler
- Interactive chat sistemi  
- Session management

---

## 🎯 Görev 5: Integration ve Test (10 puan)

### Gereksinimler:

#### 5.1. Ana Sistem Entegrasyonu
```python
class SmartCustomerSupportSystem:
    def __init__(self):
        # Tüm komponentleri birleştiren ana sınıf
        
    def handle_customer_request(self, customer_id, message):
        # Tam iş akışı implementasyonu
        # Chain -> Memory -> Tools -> Streaming
```

#### 5.2. Test Senaryoları
```python
def test_scenarios():
    # 5 farklı müşteri senaryosu
    # Her senaryo için beklenen sonuçlar
    # Performance metrics
```

### Beklenen Çıktı:
- Unified sistem sınıfı
- Comprehensive test suite
- Performance reports

---

## 📁 Dosya Yapısı

Projenizi aşağıdaki yapıda organize edin:

```
homework_solution/
├── main.py                 # Ana sistem
├── chains/
│   ├── __init__.py
│   ├── analysis_chain.py   # Müşteri analiz chain
│   ├── response_chain.py   # Yanıt üretim chain  
│   └── quality_chain.py    # Kalite kontrol chain
├── memory/
│   ├── __init__.py
│   ├── hybrid_memory.py    # Hibrit memory sistem
│   └── customer_memory.py  # Müşteri profil memory
├── tools/
│   ├── __init__.py
│   ├── ticket_tool.py      # Ticket management
│   ├── knowledge_tool.py   # Knowledge base
│   └── customer_tool.py    # Customer database
├── streaming/
│   ├── __init__.py
│   ├── handlers.py         # Streaming handlers
│   └── chat_interface.py   # Chat interface
├── tests/
│   ├── __init__.py
│   ├── test_chains.py      # Chain testleri
│   ├── test_memory.py      # Memory testleri
│   ├── test_tools.py       # Tool testleri
│   └── test_integration.py # Integration testleri
├── data/
│   ├── knowledge_base.json # Simüle bilgi tabanı
│   └── customer_data.json  # Simüle müşteri verisi
├── requirements.txt
└── README.md              # Proje açıklaması
```

---

## 🧪 Test Senaryoları

### Senaryo 1: Teknik Destek
```
Müşteri: "Uygulamanız sürekli çöküyor, nasıl çözebilirim?"
Beklenen: Teknik kategori, yüksek aciliyet, çözüm adımları
```

### Senaryo 2: Faturalandırma
```
Müşteri: "Bu ay faturamda garip bir ücret var, açıklayabilir misiniz?"
Beklenen: Billing kategori, orta aciliyet, detay sorgulaması
```

### Senaryo 3: Genel Bilgi
```
Müşteri: "Yeni özellikler ne zaman gelecek?"
Beklenen: Genel kategori, düşük aciliyet, roadmap bilgisi
```

### Senaryo 4: Kızgın Müşteri
```
Müşteri: "Bu hizmetten çok memnun değilim, iptal etmek istiyorum!"
Beklenen: Kritik aciliyet, empati odaklı yanıt, retention
```

### Senaryo 5: Takip
```
Müşteri: "Geçen hafta açtığım ticket'ın durumu nedir?"
Beklenen: Ticket sorgulama, durum güncellemesi
```

---

## 📊 Değerlendirme Kriterleri

### Code Quality (20%)
- ✅ Clean, readable kod
- ✅ Proper error handling
- ✅ Type hints kullanımı
- ✅ Documentation

### Functionality (40%)
- ✅ Chain implementasyonu (10%)
- ✅ Memory sistemi (10%)  
- ✅ Custom tools (10%)
- ✅ Streaming interface (5%)
- ✅ Integration (5%)

### Innovation (20%)
- ✅ Yaratıcı çözümler
- ✅ Ek özellikler
- ✅ Performance optimizasyonları
- ✅ User experience iyileştirmeleri

### Testing & Documentation (20%)
- ✅ Comprehensive tests
- ✅ Clear README
- ✅ Code comments
- ✅ Usage examples

---

## 🚀 Bonus Özellikler (+15 puan)

### Bonus 1: Multi-language Support (+5 puan)
```python
class MultiLanguageSupport:
    def detect_language(self, message):
        # Dil tespiti
    
    def translate_response(self, response, target_lang):
        # Yanıt çevirisi
```

### Bonus 2: Sentiment Analysis (+5 puan)
```python
class SentimentAnalyzer:
    def analyze_sentiment(self, message):
        # Müşteri duygusal durumu analizi
        # Yanıt tonunu buna göre ayarla
```

### Bonus 3: Analytics Dashboard (+5 puan)
```python
class AnalyticsDashboard:
    def generate_report(self):
        # Müşteri etkileşim raporları
        # Performance metrics
        # Trend analizi
```

---

## 📝 Teslim Gereksinimleri

### Dosya Formatı:
- **ZIP dosyası**: `hafta5_odev_[isim_soyisim].zip`
- **Klasör adı**: `homework_solution`

### İçerik:
1. ✅ Tüm kaynak kod dosyaları
2. ✅ Requirements.txt
3. ✅ README.md (kurulum + kullanım)
4. ✅ Test sonuçları (screenshots)
5. ✅ .env.example dosyası

### Çalıştırma:
```bash
cd homework_solution
pip install -r requirements.txt
python main.py
```

### Demo Video (Opsiyonel):
- 2-3 dakikalık sistem demo'su
- Temel özelliklerin gösterimi
- MP4 formatında

---

## 🕐 Teslim Tarihi

**Son teslim**: Hafta 6 dersinden önce  
**Geç teslim**: %20 puan kesintisi (günlük)

---

## 💡 İpuçları

### Development Tips:
- Küçük parçalar halinde geliştirin
- Her komponent için unit test yazın
- Git ile version control kullanın
- Code review yapın

### Testing Tips:
- Mock data kullanın
- Edge case'leri test edin
- Performance benchmark'ları alın
- Memory usage kontrolü yapın

### Documentation Tips:
- API documentation yazın
- Usage example'lar ekleyin
- Troubleshooting guide hazırlayın
- Architecture diagram çizin

---

## 🆘 Yardım ve Destek

### Office Hours:
- **Zaman**: Salı-Çarşamba 14:00-16:00
- **Platform**: Discord/Zoom
- **Soru türleri**: Teknik destek, concept açıklamaları

### FAQ:
**S: OpenAI API limitim yeterli değil**  
**C**: Mock responses kullanabilir veya daha küçük model seçebilirsiniz

**S: Import error alıyorum**  
**C**: Virtual environment aktif olduğundan emin olun

**S: Memory çok fazla RAM kullanıyor**  
**C**: ConversationSummaryMemory veya WindowMemory kullanın

### Resources:
- [LangChain Cookbook](https://github.com/langchain-ai/langchain)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

---

## 🏆 Örnek Çıktı

```bash
🤖 Smart Customer Support System Started!

📞 New Customer Request:
Customer ID: CUS001
Message: "My app keeps crashing every time I try to upload a photo"

🔍 Analysis Results:
- Category: Technical Support
- Urgency: High  
- Sentiment: Frustrated
- Language: English

💭 Generating Response...
🔧 Checking Knowledge Base...
📋 Creating Support Ticket: TKT20241013001

📝 Response:
"Hi there! I understand how frustrating app crashes can be, especially when 
you're trying to upload photos. I've found some troubleshooting steps that 
should help resolve this issue..."

✅ Quality Score: 8.7/10
🎟️ Ticket Created: TKT20241013001
⏱️ Response Time: 2.3 seconds
```

**Başarılar! 🚀**

Bu ödev, gerçek dünya senaryolarında LangChain kullanımınızı pekiştirecek ve portfolio projeniz olarak kullanabileceğiniz kapsamlı bir sistem oluşturmanızı sağlayacak.