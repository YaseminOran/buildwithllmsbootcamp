"""
Retrieval-Augmented Generation (RAG) Sistemi
===========================================

Bu kod basit bir RAG sistemini gösterir:
1. Belge veri tabanı oluşturma
2. Sorgu embedding'i çıkarma  
3. En yakın belgeyi bulma (retrieval)
4. LLM ile prompt oluşturma ve yanıt alma

RAG Süreci:
Query → Embedding → Similarity Search → Document Retrieval → Prompt + Context → LLM → Response

Gerekli Kütüphaneler:
pip install sentence-transformers openai anthropic numpy python-dotenv
"""

import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json
from dotenv import load_dotenv

# API anahtarları için
load_dotenv()

# LLM seçimi için import'lar
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

print("🤖 RAG (Retrieval-Augmented Generation) Sistemi")
print("="*60)

# Adım 1: Belge Veri Tabanı Oluşturma
print("\n📚 1. Belge Veri Tabanı Oluşturma")
print("-" * 40)

# Örnek belge koleksiyonu
documents = [
    {
        "id": "doc_1",
        "title": "Python Programlama",
        "content": """Python, yüksek seviyeli, yorumlamalı bir programlama dilidir. 
        1991 yılında Guido van Rossum tarafından geliştirilmiştir. Python'un sözdizimi 
        oldukça basit ve okunabilirdir. Web geliştirme, veri analizi, yapay zeka ve 
        bilimsel hesaplamalar için yaygın olarak kullanılır. Django ve Flask gibi 
        popüler web framework'leri vardır.""",
        "category": "Programlama"
    },
    {
        "id": "doc_2", 
        "title": "Yapay Zeka ve Machine Learning",
        "content": """Yapay zeka (AI), makinelerin insan benzeri zeka gerektiren görevleri 
        yerine getirmesi anlamına gelir. Machine learning, AI'nin bir alt dalıdır ve 
        makinelerin veriden öğrenmesini sağlar. TensorFlow, PyTorch ve Scikit-learn 
        gibi kütüphaneler ML geliştirme için kullanılır. Supervised learning, 
        unsupervised learning ve reinforcement learning temel ML türleridir.""",
        "category": "Yapay Zeka"
    },
    {
        "id": "doc_3",
        "title": "Veri Bilimi ve Analitik",
        "content": """Veri bilimi, büyük veri setlerinden anlamlı bilgiler çıkarma sanatıdır. 
        Pandas, NumPy ve Matplotlib gibi Python kütüphaneleri veri manipülasyonu ve 
        görselleştirme için kullanılır. Veri temizleme, keşif analizi, istatistiksel 
        modelleme ve makine öğrenmesi veri biliminin temel bileşenleridir. 
        İş zekası ve karar verme süreçlerinde kritik role sahiptir.""",
        "category": "Veri Bilimi"
    },
    {
        "id": "doc_4",
        "title": "Web Geliştirme",
        "content": """Web geliştirme, internet için web siteleri ve uygulamaları oluşturma 
        sürecidir. Frontend geliştirme HTML, CSS ve JavaScript kullanır. Backend 
        geliştirme için Python (Django, Flask), JavaScript (Node.js) veya diğer 
        diller kullanılabilir. Responsive tasarım, API geliştirme ve veritabanı 
        yönetimi modern web geliştirmenin temel konularıdır.""",
        "category": "Web Geliştirme"
    },
    {
        "id": "doc_5",
        "title": "Veritabanı Yönetimi",
        "content": """Veritabanı yönetim sistemleri (DBMS), verilerin organize edilmesi, 
        saklanması ve erişimi için kullanılır. SQL (Structured Query Language) 
        ilişkisel veritabanları için standart dildir. PostgreSQL, MySQL ve SQLite 
        popüler ilişkisel veritabanlarıdır. NoSQL veritabanları (MongoDB, Redis) 
        esnek veri modelleri sunar. ACID özellikleri veri tutarlılığını sağlar.""",
        "category": "Veritabanı"
    }
]

print(f"✅ {len(documents)} belge yüklendi:")
for doc in documents:
    print(f"   📄 {doc['id']}: {doc['title']} ({doc['category']})")

# Adım 2: Embedding Model Yükleme ve Belge Embedding'leri
print("\n🧠 2. Embedding Model Yükleme")
print("-" * 40)

# Sentence transformer model yükle
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"✅ Model yüklendi: {model.get_sentence_embedding_dimension()} boyutlu embedding")

# Belge içeriklerini embedding'e çevir
print("\n🔄 Belge embedding'leri oluşturuluyor...")
document_texts = [doc['content'] for doc in documents]
document_embeddings = model.encode(document_texts)

print(f"✅ {len(document_embeddings)} belge embedding'i oluşturuldu")
print(f"📊 Embedding şekli: {document_embeddings.shape}")

# Adım 3: RAG Pipeline Fonksiyonları
print("\n⚙️  3. RAG Pipeline Fonksiyonları")
print("-" * 40)

def retrieve_documents(query: str, top_k: int = 1) -> List[Tuple[Dict, float]]:
    """
    Sorgu için en yakın belgeleri bulur
    
    Args:
        query: Arama sorgusu
        top_k: Kaç belge döndürülecek
    
    Returns:
        (belge, benzerlik_skoru) tuple'ları listesi
    """
    print(f"🔍 Sorgu: '{query}'")
    
    # Sorgu embedding'ini çıkar
    query_embedding = model.encode([query])
    print(f"📊 Sorgu embedding boyutu: {query_embedding.shape}")
    
    # Cosine similarity hesapla
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    print(f"💯 Benzerlik skorları hesaplandı: {len(similarities)} belge")
    
    # En yüksek skorları bul
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    print(f"\n🎯 En yakın {top_k} belge:")
    for i, idx in enumerate(top_indices):
        doc = documents[idx]
        score = similarities[idx]
        results.append((doc, score))
        print(f"   {i+1}. {doc['title']} (Skor: {score:.4f})")
    
    return results

def create_rag_prompt(query: str, context_docs: List[Dict]) -> str:
    """
    RAG için prompt oluşturur
    
    Args:
        query: Kullanıcı sorusu
        context_docs: Bağlam belgeleri
    
    Returns:
        Oluşturulan prompt
    """
    context_text = "\n\n".join([
        f"Belge {i+1} - {doc['title']}:\n{doc['content']}"
        for i, doc in enumerate(context_docs)
    ])
    
    prompt = f"""Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Sadece verilen bağlamda yer alan bilgileri kullan.

BAĞLAM:
{context_text}

SORU: {query}

YANIT:"""
    
    return prompt

def answer_with_openai(prompt: str) -> str:
    """OpenAI ile yanıt üret"""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI kütüphanesi yüklü değil. 'pip install openai' komutu ile yükleyin."
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "❌ OPENAI_API_KEY environment variable bulunamadı. .env dosyasına ekleyin."
    
    try:
        # OpenAI client'ı doğru şekilde başlat
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen yardımcı bir AI asistanısın. Verilen bağlam bilgilerini kullanarak soruları yanıtla."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
            top_p=1.0
        )
        
        return response.choices[0].message.content.strip()
        
    except openai.AuthenticationError:
        return "❌ OpenAI API anahtarı geçersiz. Lütfen doğru API anahtarını .env dosyasına ekleyin."
    except openai.RateLimitError:
        return "❌ OpenAI API rate limit aşıldı. Lütfen daha sonra tekrar deneyin."
    except openai.APIConnectionError:
        return "❌ OpenAI API'ye bağlanılamadı. İnternet bağlantınızı kontrol edin."
    except Exception as e:
        return f"❌ OpenAI API hatası: {str(e)}"


def rag_pipeline(query: str, llm_provider: str = "mock") -> Dict:
    """
    Tam RAG pipeline'ı
    
    Args:
        query: Kullanıcı sorusu
        llm_provider: "openai" veya "mock"
    
    Returns:
        RAG sonuçları
    """
    print(f"\n🚀 RAG Pipeline Başlatılıyor...")
    print(f"🔧 LLM Provider: {llm_provider}")
    
    # 1. Retrieval - En yakın belgeyi bul
    print(f"\n📖 ADIM 1: RETRIEVAL")
    retrieved_docs = retrieve_documents(query, top_k=2)
    context_docs = [doc for doc, score in retrieved_docs]
    
    # 2. Prompt oluşturma
    print(f"\n✍️  ADIM 2: PROMPT OLUŞTURMA")
    prompt = create_rag_prompt(query, context_docs)
    print(f"📝 Prompt uzunluğu: {len(prompt)} karakter")
    
    # 3. LLM ile yanıt alma
    print(f"\n🤖 ADIM 3: LLM YANITI")
    if llm_provider == "openai":
        response = answer_with_openai(prompt)
    else:
        # Mock yanıt
        response = f"""Bu bir mock yanıttır. Gerçek RAG sistemi şu adımları tamamladı:

1. ✅ Sorgu embedding'i oluşturuldu
2. ✅ En yakın belgeler bulundu:
   - {context_docs[0]['title']} (Skor: {retrieved_docs[0][1]:.4f})
   - {context_docs[1]['title']} (Skor: {retrieved_docs[1][1]:.4f})
3. ✅ Prompt oluşturuldu ({len(prompt)} karakter)
4. 🔄 LLM yanıtı bekleniyor...

Gerçek LLM kullanmak için API anahtarınızı .env dosyasına ekleyin:
- OpenAI: OPENAI_API_KEY=sk-your-key-here"""
    
    return {
        'query': query,
        'retrieved_docs': retrieved_docs,
        'prompt': prompt,
        'response': response,
        'llm_provider': llm_provider
    }

# Adım 4: RAG Sistemini Test Etme
print("\n🧪 4. RAG Sistemini Test Etme")
print("-" * 40)

# Test sorguları
test_queries = [
    "Python nedir ve ne için kullanılır?",
    "Machine learning türleri nelerdir?",
    "Veri bilimi için hangi Python kütüphaneleri kullanılır?",
    "Web geliştirmede frontend ve backend arasındaki fark nedir?",
    "SQL nedir?"
]

print(f"📋 {len(test_queries)} test sorusu hazırlandı:")
for i, query in enumerate(test_queries, 1):
    print(f"   {i}. {query}")

# İlk soruyu detaylı test et
print(f"\n🔍 DETAYLI TEST - İlk Sorgu")
print("="*50)

test_query = test_queries[0]
result = rag_pipeline(test_query, llm_provider="mock")

print(f"\n📊 RAG SONUÇLARI:")
print(f"🔤 Sorgu: {result['query']}")
print(f"\n📚 Bulunan Belgeler:")
for i, (doc, score) in enumerate(result['retrieved_docs'], 1):
    print(f"   {i}. {doc['title']} - Skor: {score:.4f}")
    print(f"      Kategori: {doc['category']}")
    print(f"      İçerik: {doc['content'][:100]}...")

print(f"\n📝 Oluşturulan Prompt (ilk 200 karakter):")
print(f"'{result['prompt'][:200]}...'")

print(f"\n🤖 LLM Yanıtı:")
print(result['response'])

# Adım 5: Tüm sorguları hızlı test
print(f"\n⚡ 5. Hızlı Test - Tüm Sorgular")
print("-" * 40)

for i, query in enumerate(test_queries, 1):
    print(f"\n📋 Test {i}: {query}")
    retrieved = retrieve_documents(query, top_k=1)
    best_doc, score = retrieved[0]
    print(f"   🎯 En iyi eşleşme: {best_doc['title']} (Skor: {score:.4f})")

# Adım 6: RAG Sistem Metrikleri
print(f"\n📈 6. RAG Sistem Metrikleri")
print("-" * 40)

# Embedding kalitesi analizi
all_similarities = []
for query in test_queries:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    all_similarities.extend(similarities)

avg_similarity = np.mean(all_similarities)
max_similarity = np.max(all_similarities)
min_similarity = np.min(all_similarities)

print(f"🎯 Embedding Performansı:")
print(f"   Ortalama benzerlik: {avg_similarity:.4f}")
print(f"   Maksimum benzerlik: {max_similarity:.4f}")
print(f"   Minimum benzerlik: {min_similarity:.4f}")

print(f"\n🔧 Sistem Özellikleri:")
print(f"   📚 Toplam belge sayısı: {len(documents)}")
print(f"   🧠 Embedding boyutu: {document_embeddings.shape[1]}")
print(f"   📊 Model: {model.get_sentence_embedding_dimension()}D sentence-transformer")

# Adım 7: RAG İyileştirme Önerileri
print(f"\n💡 7. RAG Sistem İyileştirme Önerileri")
print("-" * 40)

improvement_tips = """
🚀 PERFORMANS İYİLEŞTİRMELERİ:

1. 📚 Veri Kalitesi:
   • Belgeleri daha küçük chunk'lara böl
   • Metadata ekle (kategori, tarih, kaynak)
   • Overlapping chunk'lar kullan

2. 🔍 Retrieval İyileştirme:
   • Hybrid search (semantic + keyword)
   • Re-ranking modelleri kullan
   • Query expansion uygula

3. 🤖 LLM Optimizasyonu:
   • Prompt engineering
   • Few-shot examples ekle
   • Response validation

4. 📊 Değerlendirme:
   • BLEU, ROUGE skorları
   • Human evaluation
   • A/B testing

5. 🏗️ Mimari İyileştirmeler:
   • Vector database kullan (Pinecone, Weaviate)
   • Caching stratejileri
   • Async processing
"""

print(improvement_tips)

print(f"\n✅ RAG sistemi demonstrasyonu tamamlandı!")

print(f"📄 .env dosyanızda OPENAI_API_KEY değişkenini ayarlayarak gerçek LLM kullanabilirsiniz")