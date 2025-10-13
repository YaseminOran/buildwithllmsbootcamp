# 🚀 Build with LLMs Bootcamp

Bu bootcamp, Large Language Models (LLM) teknolojisini kullanarak pratik uygulamalar geliştirmeyi öğreten kapsamlı bir eğitim programıdır. 8 haftalık yoğun program boyunca, LLM'lerin temellerinden başlayarak, gerçek dünya uygulamalarına kadar olan tüm konuları ele alacaksınız.

## 📚 Program İçeriği

### [Modül 1: LLM Temelleri ve Python ile NLP'ye Giriş](./hafta_1/)
- **Süre**: 1 Hafta
- **Konular**:
  - Large Language Models nedir ve nasıl çalışır?
  - NLP temel kavramları ve Python ile uygulama
  - Tokenization, encoding/decoding işlemleri
  - Transformer mimarisi temelleri
  - Hugging Face ekosistemi tanıtımı
- **Pratik Projeler**: Temel NLP görevleri, text preprocessing, basit model kullanımı

### [Modül 2: Prompt Engineering ve API Tabanlı Kullanım](./hafta_2/)
- **Süre**: 1 Hafta
- **Konular**:
  - Zero-shot, Few-shot ve Chain of Thought prompting
  - Role-based prompt yazım teknikleri
  - OpenAI API kullanımı (ChatCompletion, Function Calling)
  - Prompt optimizasyon stratejileri
  - API güvenliği ve rate limiting
- **Pratik Projeler**: Akıllı chatbot sistemi, function calling uygulamaları

### [Modül 3: AutoTokenizer, AutoModel ve Pipeline Optimizasyonu](./hafta_3/)
- **Süre**: 1 Hafta
- **Konular**:
  - AutoTokenizer & AutoModel yapısı ve pipeline kullanımı
  - GPT, BERT ve T5 modellerinin karşılaştırması
  - CPU/GPU performans yönetimi ve model optimizasyonu
  - Pipeline ile performans ölçümü ve kıyaslama
  - Model quantization ve batch processing
- **Pratik Projeler**: Model karşılaştırma benchmark'ı, performans optimizasyon araçları

### [Modül 4: Embedding, Vector Database ve Semantic Search](./hafta_4/)
- **Süre**: 1 Hafta
- **Konular**:
  - Text embedding'lerin teorisi ve pratiği
  - Vector database sistemleri (Pinecone, Weaviate, Chroma)
  - Semantic search ve similarity hesaplamaları
  - Retrieval Augmented Generation (RAG) temelleri
  - Embedding modelleri karşılaştırması
- **Pratik Projeler**: Semantic search motoru, RAG tabanlı Q&A sistemi

### [Modül 5: İleri Düzey LangChain - Chain, Memory, Tools ve Streaming](./hafta_5/)
- **Süre**: 1 Hafta
- **Konular**:
  - Chain yapıları (LLMChain, SequentialChain, Custom Chains)
  - Memory yönetimi (Buffer, Window, Summary, Hybrid Memory)
  - Tool integration ve Agent'lar (Custom Tools, ReAct Agents)
  - Senaryo bazlı uygulamalar (Müşteri hizmetleri, İçerik oluşturma)
  - Streaming output ve canlı veri akışı (Real-time responses)
- **Pratik Projeler**: Akıllı müşteri destek sistemi, streaming chat uygulaması

### [Modül 6: Fine-Tuning ve Hafif Model Eğitimi](./hafta_6/)
- **Süre**: 1 Hafta
- **Konular**:
  - Transfer learning ve domain adaptation
  - LoRA ve QLoRA teknikleri
  - Dataset hazırlama ve augmentation
  - Training pipeline'ları
  - Model evaluation ve metrics
- **Pratik Projeler**: Domain-specific model fine-tuning, instruction tuning

### [Modül 7: LLM Tabanlı Uygulama Dağıtımı](./hafta_7/)
- **Süre**: 1 Hafta
- **Konular**:
  - Production deployment stratejileri
  - Model optimization ve quantization
  - API gateway ve microservices
  - Monitoring ve logging
  - Scaling ve performance optimization
- **Pratik Projeler**: Production-ready LLM uygulaması deployment

### [Modül 8: LLM Protokolleri ile Sistem Mimarisi](./hafta_8/)
- **Süre**: 1 Hafta
- **Konular**:
  - Enterprise LLM mimarileri
  - Multi-model orchestration
  - Security ve privacy considerations
  - Cost optimization stratejileri
  - Future trends ve emerging technologies
- **Pratik Projeler**: Kapsamlı LLM sistemi tasarımı ve implementasyonu

## 🎯 Program Hedefleri

Bu bootcamp sonunda katılımcılar:

✅ **LLM Teknolojilerini** derinlemesine anlayacak
✅ **Production-ready uygulamalar** geliştirebilecek
✅ **Modern AI tools** ve framework'leri etkin kullanabilecek
✅ **End-to-end LLM projeleri** yönetebilecek
✅ **Industry best practices** uygulayabilecek

## 🔧 Teknoloji Stack'i

### Core Technologies
- **Python** - Ana programlama dili
- **PyTorch** - Deep learning framework (Hafta 3+)
- **Transformers** - Hugging Face model kütüphanesi
- **OpenAI API** - LLM servisleri
- **LangChain** - LLM uygulama framework'ü

### Databases & Vector Stores
- **Pinecone/Chroma** - Vector database
- **PostgreSQL** - İlişkisel database
- **Redis** - Caching ve session management

### Deployment & Infrastructure
- **Docker** - Containerization
- **FastAPI** - API development
- **Streamlit** - Rapid prototyping
- **AWS/Azure** - Cloud deployment

## 📋 Ön Koşullar

### Gerekli Bilgiler
- **Python** programlama (orta seviye)
- **Git** ve version control
- **REST API** temel bilgisi
- **Linux/Unix** command line kullanımı (macOS/Linux)
- **Command Prompt/PowerShell** kullanımı (Windows)

### Önerilen Bilgiler
- Machine Learning temel kavramları
- Deep Learning temelleri
- Cloud services deneyimi
- Docker kullanımı

## 🚦 Başlangıç

### 1. Repository'yi Clone Edin
```bash
git clone https://github.com/YaseminOran/buildwithllmsbootcamp.git
cd buildwithllmsbootcamp
```

### 2. Python Environment Hazırlayın
```bash
# Python 3.8+ gerekli
python -m venv bootcamp_env
source bootcamp_env/bin/activate  # Linux/Mac
# bootcamp_env\Scripts\activate  # Windows
```

### 3. Hafta 3 Hızlı Başlangıç
```bash
cd hafta_3

# macOS/Linux
./start.sh

# Windows
start.bat
```

### 4. API Keys Hazırlayın
- OpenAI API Key (Hafta 2 için)
- Hugging Face Token (Hafta 3+ için)
- Pinecone API Key (Hafta 4 için)

## 📁 Proje Yapısı

```
buildwithllmsbootcamp/
├── README.md                 # Bu dosya
├── .gitignore               # Git ignore kuralları
├── hafta_1/                 # Modül 1: LLM Temelleri
│   ├── turkish_simple.py
│   ├── microsoft.py
│   ├── qwen.py
│   └── llm_1/ (venv)
├── hafta_2/                 # Modül 2: Prompt Engineering
│   ├── README.md
│   ├── requirements.txt
│   ├── 01_zero_shot.py
│   ├── 02_few_shot.py
│   ├── 03_chain_of_thought.py
│   ├── 04_role_based.py
│   ├── 05_chatcompletion_api.py
│   ├── 06_function_calling.py
│   ├── 07_chatbot_with_functions.py
│   ├── 08_simple_chatbot.py
│   ├── 09_web_chatbot.py
│   └── prompt/ (venv)
├── hafta_3/                 # Modül 3: Pipeline Optimizasyonu
│   ├── README.md            # Modül açıklaması
│   ├── SETUP.md             # Detaylı kurulum kılavuzu
│   ├── requirements.txt     # Python bağımlılıkları
│   ├── start.sh             # macOS/Linux kurulum scripti
│   ├── start.bat            # Windows kurulum scripti
│   ├── .gitignore           # Hafta 3 özel ignore kuralları
│   ├── 01_autotokenizer_automodel.py
│   ├── 02_gpt_bert_t5_comparison.py
│   ├── 03_cpu_gpu_optimization.py
│   ├── 04_performance_measurement.py
│   └── llm_bootcamp_env/ (venv)
├── hafta_4/                 # Modül 4: Vector Search & RAG
│   ├── README.md
│   ├── requirements.txt
│   ├── simple_rag_demo.py
│   ├── chroma_vector_search.py
│   ├── homework.md
│   └── images/
└── hafta_5/                 # Modül 5: İleri LangChain
    ├── README.md            # Modül açıklaması
    ├── requirements.txt     # Python bağımlılıkları
    ├── setup_venv.py        # Otomatik kurulum scripti
    ├── test_installation.py # Kurulum test scripti
    ├── 1_chains_basic.py    # Chain yapıları
    ├── 2_memory_examples.py # Memory yönetimi
    ├── 3_tools_and_agents.py # Tools ve Agents
    ├── 4_scenario_applications.py # Senaryo uygulamaları
    ├── 5_streaming_examples.py # Streaming output
    └── homework.md          # Hafta 5 ödev
```

## 🎓 Değerlendirme

Her modül sonunda:
- **Praktik projeler** (70%)
- **Kod kalitesi** (20%)
- **Dokümantasyon** (10%)

Final proje: Kapsamlı LLM uygulaması geliştirme

## 📞 İletişim ve Destek

- **GitHub Issues**: Teknik sorular için
- **Discussions**: Genel tartışmalar için
- **Wiki**: Detaylı dokümantasyon

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🤝 Katkıda Bulunma

Katkılarınızı memnuniyetle karşılıyoruz! Lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

---

**🚀 Build with LLMs Bootcamp - Future of AI Development Starts Here!**

*Son güncelleme: Eylül 2024*

## 🎯 Hafta 3 Özel Notları

### Cross-Platform Desteği
Hafta 3 modülü tüm işletim sistemlerinde çalışır:
- **macOS**: Native MPS (Apple Silicon) desteği
- **Linux**: CUDA GPU desteği 
- **Windows**: CUDA GPU ve CPU desteği

### Sistem Gereksinimleri
- **Minimum**: 8 GB RAM, Python 3.8+
- **Önerilen**: 16 GB RAM, GPU (CUDA/MPS)
- **Disk**: 10 GB boş alan (model cache için)

### Hızlı Kurulum
```bash
cd hafta_3
chmod +x start.sh  # sadece ilk seferde
./start.sh         # macOS/Linux
# veya
start.bat          # Windows
```