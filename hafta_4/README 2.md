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
- Document retrieval ve embedding search
- LLM integration (OpenAI/Claude)
- Tam RAG pipeline implementasyonu

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
- `rag_system.py` - Tam özellikli RAG sistemi
- `simple_rag_demo.py` - Basit RAG demonstrasyonu
- `requirements.txt` - Gerekli Python kütüphaneleri
- `.env` - API anahtarları için örnek dosya

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

### Basit RAG Demo
```bash
python simple_rag_demo.py
```

### Tam RAG Sistemi
```bash
# Önce API anahtarlarınızı ayarlayın
cp .env.example .env
# .env dosyasını düzenleyip API anahtarlarınızı ekleyin
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

### 5. Basit RAG Demo
- 3 kısa belge ile knowledge base
- Query → Embedding → Similarity → Retrieval
- Mock LLM yanıtı ile RAG pipeline
- Adım adım açıklamalı süreç

### 6. Tam RAG Sistemi
- 5 belge ile zengin knowledge base
- OpenAI ve Claude API entegrasyonu
- Çoklu belge retrieval
- Production-ready RAG pipeline

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

RAG sistemi, büyük dil modellerinin bilgi eksikliğini gidermek için geliştirilmiş bir tekniktir:

### RAG Süreci
1. **Query** → Kullanıcı sorusu
2. **Embedding** → Sorguyu vektöre çevir
3. **Retrieval** → En yakın belgeleri bul
4. **Augmentation** → Bağlamı prompt'a ekle
5. **Generation** → LLM ile yanıt üret

### Avantajları
- ✅ Güncel bilgi erişimi
- ✅ Domain-specific knowledge
- ✅ Hallucination azaltma
- ✅ Şeffaflık (kaynak gösterme)

### API Kurulumu
```bash
# .env dosyasını oluşturun
cp .env.example .env

# API anahtarlarınızı ekleyin:
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_claude_key_here
```