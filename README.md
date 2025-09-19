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

### [Modül 3: Hugging Face Transformers Derinlemesine](./hafta_3/)
- **Süre**: 1 Hafta
- **Konular**:
  - Transformers kütüphanesi detaylı kullanımı
  - Pre-trained modellerin yüklenmesi ve kullanımı
  - Pipeline'lar ve custom task'lar
  - Model hub ve community modelleri
  - Tokenizer'lar ve özel tokenization
- **Pratik Projeler**: Multi-task NLP uygulaması, custom pipeline geliştirme

### [Modül 4: Embedding, Vector Database ve Semantic Search](./hafta_4/)
- **Süre**: 1 Hafta
- **Konular**:
  - Text embedding'lerin teorisi ve pratiği
  - Vector database sistemleri (Pinecone, Weaviate, Chroma)
  - Semantic search ve similarity hesaplamaları
  - Retrieval Augmented Generation (RAG) temelleri
  - Embedding modelleri karşılaştırması
- **Pratik Projeler**: Semantic search motoru, RAG tabanlı Q&A sistemi

### [Modül 5: LangChain ile Çok Adımlı Uygulama Geliştirme](./hafta_5/)
- **Süre**: 1 Hafta
- **Konular**:
  - LangChain framework'ü derinlemesine
  - Chain'ler ve Agent'lar
  - Memory yönetimi ve conversation handling
  - Tool integration ve custom tools
  - Multi-agent sistemler
- **Pratik Projeler**: AI asistan uygulaması, document analysis sistemi

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
- **OpenAI API** - LLM servisleri
- **Hugging Face** - Model hub ve araçlar
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
- **Linux/Unix** command line kullanımı

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

### 3. Her Modül İçin Ayrı Environment
Her hafta kendi sanal ortamına sahiptir. Detaylar için ilgili hafta klasörüne bakın.

### 4. API Keys Hazırlayın
- OpenAI API Key
- Hugging Face Token
- Pinecone API Key (4. hafta için)

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
└── hafta_2/                 # Modül 2: Prompt Engineering
    ├── README.md
    ├── requirements.txt
    ├── 01_zero_shot.py
    ├── 02_few_shot.py
    ├── 03_chain_of_thought.py
    ├── 04_role_based.py
    ├── 05_chatcompletion_api.py
    ├── 06_function_calling.py
    ├── 07_chatbot_with_functions.py
    ├── 08_simple_chatbot.py
    ├── 09_web_chatbot.py
    └── prompt/ (venv)
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