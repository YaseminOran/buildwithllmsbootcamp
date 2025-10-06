# Hafta 4: Embedding ve Vektör Arama

Bu öğretici, Python'da embedding çıkarma ve vektör arama teknolojilerini kapsamlı olarak ele alır.

## 🎯 Ne Öğreneceksiniz?

### Bölüm 1: Temel Embedding'ler
- **Embedding nedir?** - Metinlerin sayısal vektör temsilleri
- **Sentence-transformers** kullanımı
- **Cosine similarity** hesaplama
- **TSNE** ile 2D görselleştirme

### Bölüm 2: FAISS ile Vektör Arama
- **FAISS** (Facebook AI Similarity Search) kullanımı
- Farklı index türleri (Flat, IVF, HNSW)
- 512 boyutlu vektörlerle en yakın komşu arama
- Performans optimizasyonu

### Bölüm 3: Chroma DB ile Vektör Arama
- **Chroma DB** kurulumu ve kullanımı
- Metadata ile zenginleştirilmiş arama
- Filtreleme ve sorgu optimizasyonu
- Persistence ve koleksiyon yönetimi

### Bölüm 4: FAISS vs Chroma Karşılaştırması
- Performans benchmarking
- Hız, bellek, ölçeklenebilirlik analizi
- Kullanım senaryoları ve öneriler

### Bölüm 5: RAG (Retrieval-Augmented Generation)
- **RAG sistemi** nedir ve nasıl çalışır
- ChromaDB vector database entegrasyonu
- Document retrieval ve semantic search
- OpenAI GPT integration
- Production-ready RAG pipeline

## 🚀 Kurulum

```bash
# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

## 📁 Dosyalar

- `embedding_tutorial.py` - Temel embedding öğreticisi
- `faiss_vector_search.py` - FAISS vektör arama örnekleri
- `chroma_vector_search.py` - Chroma DB örnekleri
- `performance_comparison.py` - FAISS vs Chroma karşılaştırması
- `rag_system.py` - Tam özellikli RAG sistemi (5 belge, detaylı analiz)
- `simple_rag_demo.py` - Modern RAG demo (ChromaDB + OpenAI)
- `requirements.txt` - Gerekli Python kütüphaneleri
- `.env` - API anahtarları dosyası

## 🏃‍♂️ Çalıştırma

### Temel Embedding Öğreticisi
```bash
cd hafta_4
python embedding_tutorial.py
```

### FAISS Vektör Arama
```bash
python faiss_vector_search.py
```

### Chroma DB Vektör Arama
```bash
python chroma_vector_search.py
```

### Performans Karşılaştırması
```bash
python performance_comparison.py
```

### Modern RAG Demo (ChromaDB + OpenAI)
```bash
python simple_rag_demo.py
```

### Tam RAG Sistemi (Kapsamlı Analiz)
```bash
# API anahtarınız varsa .env dosyasına ekleyin
python rag_system.py
```

## 📊 Öğretici İçerikleri

### 1. Embedding Tutorial
- 10 farklı Türkçe cümleden embedding çıkarma
- Cosine similarity hesaplama
- En benzer cümle çiftini bulma
- TSNE ile 2D görselleştirme

### 2. FAISS Tutorial
- 10,000 adet 512 boyutlu rastgele vektör
- Flat ve IVF index karşılaştırması
- Performance benchmarking
- Bellek kullanımı analizi

### 3. Chroma Tutorial
- 1,000 vektör ile koleksiyon oluşturma
- Metadata ile kategori bazlı filtreleme
- Kompleks sorgular ve filtreleme
- Koleksiyon istatistikleri

### 4. Performans Karşılaştırması
- Farklı boyutlarda (128, 256, 512) test
- Farklı vektör sayılarında (1K, 5K, 10K) test
- Hız, bellek, throughput analizi
- Detaylı görselleştirme

### 5. Modern RAG Demo
- 3 belge ile ChromaDB vector database
- Query → Vector Search → Retrieval → LLM
- OpenAI GPT entegrasyonu (.env desteği)
- Production-ready RAG pipeline

### 6. Tam RAG Sistemi
- 5 belge ile kapsamlı knowledge base
- Sentence transformers + manual similarity
- OpenAI API entegrasyonu
- Detaylı performans analizi ve öneriler

## 🔍 Çıktı Dosyaları

Program çalıştırıldığında üretilen dosyalar:
- `embedding_visualization.png` - TSNE görselleştirmesi
- `faiss_performance.png` - FAISS performans grafikleri
- `faiss_vs_chroma_comparison.png` - Kapsamlı karşılaştırma

## ⚖️ FAISS vs Chroma Özet

| Özellik | FAISS | Chroma DB |
|---------|-------|-----------|
| **Hız** | Çok hızlı (C++ backend) | Orta hızlı |
| **Kullanım** | Teknik bilgi gerekli | Kolay API |
| **Ölçeklenebilirlik** | Milyarlarca vektör | Milyonlarca vektör |
| **Metadata** | Yok | Zengin destek |
| **GPU** | Mükemmel | Sınırlı |
| **Use Case** | Büyük ölçek, performans | Prototip, web app |

## 🤖 RAG (Retrieval-Augmented Generation)

RAG sistemi, LLM'lerin bilgi eksikliğini vector database ile gidermek için geliştirilmiş modern bir tekniktir:

### Modern RAG Pipeline
1. **Documents** → Vector Database'e yükleme (ChromaDB)
2. **Query** → Semantic search ile en yakın belgeler
3. **Retrieval** → Similarity skorları ile ranking
4. **Augmentation** → Context'i prompt'a ekleme
5. **Generation** → LLM ile contextual response

### Teknoloji Stack'i
- **Vector DB**: ChromaDB (cosine similarity)
- **Embeddings**: Sentence Transformers
- **LLM**: OpenAI GPT-3.5-turbo
- **Orchestration**: Python + dotenv

### Avantajları
- ✅ Gerçek zamanlı bilgi erişimi
- ✅ Domain-specific knowledge base
- ✅ Hallucination azaltma
- ✅ Kaynak transparency
- ✅ Production-ready scalability

### API Kurulumu
```bash
# .env dosyasına API anahtarınızı ekleyin:
OPENAI_API_KEY=sk-your-openai-key-here
```

### İki RAG Implementation
- **simple_rag_demo.py**: Modern vector DB yaklaşımı
- **rag_system.py**: Kapsamlı analiz ve karşılaştırma