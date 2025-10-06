"""
Chroma DB ile Vektör Arama
==========================

Bu kod Chroma DB kullanarak:
1. Vektör koleksiyonu oluşturma
2. Metadata ile birlikte vektör saklama
3. Semantik arama yapma

Chroma DB Avantajları:
- Kolay kullanım (high-level API)
- Metadata desteği
- Otomatik persistence
- Gömülü veritabanı

Gerekli Kütüphaneler:
pip install chromadb numpy
"""

import chromadb
import numpy as np
import time
from typing import List, Dict

print("🎨 Chroma DB ile Vektör Arama Öğreticisi")
print("="*50)

# Adım 1: Chroma Client ve Collection oluşturma
print("\n📚 1. Chroma Client ve Collection Oluşturma")
print("-" * 40)

# Client oluştur (in-memory için)
client = chromadb.Client()

# Collection oluştur
collection_name = "vector_search_demo"
try:
    # Eğer koleksiyon varsa sil
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}  # Cosine similarity kullan
)

print(f"✅ Collection oluşturuldu: {collection_name}")
print(f"📊 Similarity metric: cosine")

# Adım 2: Örnek vektörler ve metadata oluşturma
print("\n🗂️  2. Vektörler ve Metadata Hazırlama")
print("-" * 40)

# Parametreler
dimension = 512
n_vectors = 1000

# Rastgele vektörler oluştur
np.random.seed(42)
vectors = np.random.random((n_vectors, dimension)).astype('float32').tolist()

# ID'ler oluştur
ids = [f"vec_{i}" for i in range(n_vectors)]

# Metadata oluştur (kategoriler, etiketler vs.)
categories = ["teknoloji", "spor", "sanat", "bilim", "müzik"]
metadatas = []
documents = []

for i in range(n_vectors):
    category = categories[i % len(categories)]
    metadatas.append({
        "category": category,
        "index": i,
        "group": f"group_{i // 100}"
    })
    documents.append(f"Bu {category} kategorisinden örnek belge {i}")

print(f"✅ {n_vectors} vektör hazırlandı")
print(f"📝 Kategoriler: {categories}")
print(f"🏷️  Her vektör için metadata ve doküman oluşturuldu")

# Adım 3: Vektörleri Chroma'ya ekleme
print("\n💾 3. Vektörleri Chroma'ya Ekleme")
print("-" * 40)

start_time = time.time()

# Batch olarak ekle (performans için)
batch_size = 100
for i in range(0, n_vectors, batch_size):
    end_idx = min(i + batch_size, n_vectors)
    
    collection.add(
        embeddings=vectors[i:end_idx],
        metadatas=metadatas[i:end_idx],
        documents=documents[i:end_idx],
        ids=ids[i:end_idx]
    )

add_time = time.time() - start_time
print(f"✅ {collection.count()} vektör eklendi")
print(f"⏱️  Ekleme süresi: {add_time:.4f} saniye")

# Adım 4: Basit vektör arama
print("\n🔍 4. Vektör Arama İşlemleri")
print("-" * 40)

# Sorgu vektörü oluştur
query_vector = np.random.random(dimension).astype('float32').tolist()

# En yakın 5 komşuyu bul
start_time = time.time()
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)
search_time = time.time() - start_time

print(f"🎯 Arama süresi: {search_time:.4f} saniye")
print(f"📊 Bulunan sonuç sayısı: {len(results['ids'][0])}")

print("\n📋 Arama Sonuçları:")
for i, (doc_id, distance, metadata, document) in enumerate(zip(
    results['ids'][0],
    results['distances'][0],
    results['metadatas'][0],
    results['documents'][0]
)):
    print(f"  {i+1}. ID: {doc_id}")
    print(f"     Mesafe: {distance:.4f}")
    print(f"     Kategori: {metadata['category']}")
    print(f"     Grup: {metadata['group']}")
    print(f"     Doküman: {document[:50]}...")
    print()

# Adım 5: Metadata ile filtreleme
print("\n🎛️  5. Metadata ile Filtreleme")
print("-" * 40)

# Sadece "teknoloji" kategorisinde ara
tech_results = collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    where={"category": "teknoloji"}
)

print("🔬 Sadece 'teknoloji' kategorisindeki sonuçlar:")
for i, (doc_id, distance, metadata) in enumerate(zip(
    tech_results['ids'][0],
    tech_results['distances'][0],
    tech_results['metadatas'][0]
)):
    print(f"  {i+1}. ID: {doc_id}, Mesafe: {distance:.4f}, Kategori: {metadata['category']}")

# Composite filter örneği
complex_results = collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    where={
        "$and": [
            {"category": {"$in": ["teknoloji", "bilim"]}},
            {"index": {"$gte": 100}}
        ]
    }
)

print("\n🧪 Karmaşık filtre (teknoloji VEYA bilim VE index >= 100):")
for i, (doc_id, distance, metadata) in enumerate(zip(
    complex_results['ids'][0],
    complex_results['distances'][0],
    complex_results['metadatas'][0]
)):
    print(f"  {i+1}. ID: {doc_id}, Mesafe: {distance:.4f}")
    print(f"     Kategori: {metadata['category']}, Index: {metadata['index']}")

# Adım 6: Koleksiyon istatistikleri
print("\n📈 6. Koleksiyon İstatistikleri")
print("-" * 40)

print(f"📊 Toplam vektör sayısı: {collection.count()}")

# Kategorilere göre dağılım
category_counts = {}
all_metadatas = collection.get(include=['metadatas'])['metadatas']
for metadata in all_metadatas:
    category = metadata['category']
    category_counts[category] = category_counts.get(category, 0) + 1

print("\n📈 Kategori Dağılımı:")
for category, count in category_counts.items():
    print(f"  {category}: {count} vektör")

# Adım 7: FAISS vs Chroma karşılaştırması
print("\n⚖️  7. FAISS vs Chroma DB Karşılaştırması")
print("-" * 40)

comparison = """
🏃‍♂️ HIZ:
• FAISS: Çok hızlı (C++ backend)
• Chroma: Orta hızlı (Python overhead)

🛠️  KULLANIM KOLAYLIĞI:
• FAISS: Düşük seviye, teknik bilgi gerekli
• Chroma: Yüksek seviye, kolay kullanım

📏 ÖLÇEKLENEBİLİRLİK:
• FAISS: Milyarlarca vektör
• Chroma: Milyonlarca vektör

🎯 ÖZELLIKLER:
• FAISS: Sadece vektör arama
• Chroma: Metadata, persistence, API

💾 BELLEK:
• FAISS: Manuel yönetim, optimize
• Chroma: Otomatik yönetim

🔧 GPU DESTEĞİ:
• FAISS: Mükemmel GPU desteği
• Chroma: Sınırlı GPU desteği

✅ NE ZAMAN KULLAN:

FAISS:
• Çok büyük veri setleri (>10M vektör)
• Maksimum hız gerekli
• GPU kullanımı kritik
• Bare-metal performans

Chroma:
• Hızlı prototipleme
• Metadata ile zengin arama
• Kolay deployment
• Web uygulamaları
"""

print(comparison)

print("\n✅ Chroma DB öğreticisi tamamlandı!")
print(f"🗄️  Koleksiyon: {collection.count()} vektör içeriyor")