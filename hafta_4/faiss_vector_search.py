"""
FAISS ve Vektör Arama
=====================

Bu kod FAISS (Facebook AI Similarity Search) kullanarak:
1. 512 boyutlu rastgele vektörlerden index oluşturma
2. En yakın komşu arama (k-NN search)
3. Performans ölçümü

FAISS Avantajları:
- Çok hızlı (C++ backend)
- GPU desteği
- Büyük veri setleri için optimize
- Düşük seviye kontrol

Gerekli Kütüphaneler:
pip install faiss-cpu numpy matplotlib
"""

import numpy as np
import faiss
import time
import matplotlib.pyplot as plt

print("🚀 FAISS ile Vektör Arama Öğreticisi")
print("="*50)

# Adım 1: Rastgele vektör veri seti oluşturma
print("\n📊 1. Vektör Veri Seti Oluşturma")
print("-" * 30)

# Parametreler
dimension = 512          # Vektör boyutu
n_vectors = 10000       # Toplam vektör sayısı
n_query = 5            # Sorgu vektör sayısı
k = 3                  # En yakın k komşu

print(f"• Vektör boyutu: {dimension}")
print(f"• Toplam vektör sayısı: {n_vectors}")
print(f"• Sorgu sayısı: {n_query}")
print(f"• Aranacak komşu sayısı: {k}")

# Rastgele vektörler oluştur (L2 normalize edilmiş)
np.random.seed(42)
vectors = np.random.random((n_vectors, dimension)).astype('float32')
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # L2 normalize

query_vectors = np.random.random((n_query, dimension)).astype('float32')
query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

print(f"✅ Vektörler oluşturuldu: {vectors.shape}")
print(f"✅ Sorgu vektörleri: {query_vectors.shape}")

# Adım 2: FAISS Index oluşturma ve vektörleri ekleme
print("\n🔧 2. FAISS Index Oluşturma")
print("-" * 30)

# Farklı index türleri deneyelim
index_types = {
    'Flat': faiss.IndexFlatIP,      # Brute force (tam doğruluk)
    'IVF': lambda d: faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 100)  # Hızlı yaklaşık
}

results = {}

for index_name, index_creator in index_types.items():
    print(f"\n🏗️  {index_name} Index oluşturuluyor...")
    
    # Index oluştur
    if index_name == 'Flat':
        index = index_creator(dimension)
    else:
        index = index_creator(dimension)
        # IVF için training gerekli
        index.train(vectors)
    
    # Vektörleri indexe ekle
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    
    print(f"   ✅ {index.ntotal} vektör eklendi")
    print(f"   ⏱️  Ekleme süresi: {add_time:.4f} saniye")
    
    # Arama performansını test et
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time
    
    print(f"   🔍 Arama süresi: {search_time:.4f} saniye")
    print(f"   📈 Saniyede sorgu: {n_query/search_time:.0f}")
    
    results[index_name] = {
        'add_time': add_time,
        'search_time': search_time,
        'distances': distances,
        'indices': indices
    }

# Adım 3: Arama sonuçlarını analiz etme
print("\n🔍 3. Arama Sonuçları Analizi")
print("-" * 30)

for i, query_vector in enumerate(query_vectors):
    print(f"\n📍 Sorgu {i+1} için sonuçlar:")
    
    for index_name in results:
        distances = results[index_name]['distances'][i]
        indices = results[index_name]['indices'][i]
        
        print(f"  {index_name} Index:")
        for j, (dist, idx) in enumerate(zip(distances, indices)):
            print(f"    {j+1}. En yakın: Index {idx}, Mesafe: {dist:.4f}")

# Adım 4: Performans karşılaştırması görselleştirme
print("\n📊 4. Performans Görselleştirmesi")
print("-" * 30)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Ekleme süresi karşılaştırması
index_names = list(results.keys())
add_times = [results[name]['add_time'] for name in index_names]
search_times = [results[name]['search_time'] for name in index_names]

ax1.bar(index_names, add_times, color=['blue', 'red'], alpha=0.7)
ax1.set_title('Index Oluşturma Süresi', fontsize=14, fontweight='bold')
ax1.set_ylabel('Süre (saniye)', fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.bar(index_names, search_times, color=['green', 'orange'], alpha=0.7)
ax2.set_title('Arama Süresi', fontsize=14, fontweight='bold')
ax2.set_ylabel('Süre (saniye)', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/yaseminarslan/Desktop/buildwithllmsbootcamp/hafta_4/images/faiss_performance.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Adım 5: FAISS Özellikleri ve İpuçları
print("\n💡 5. FAISS İpuçları ve Özellikler")
print("-" * 30)

print("""
🎯 FAISS Index Türleri:
• IndexFlatIP: Brute force, tam doğruluk, yavaş
• IndexIVFFlat: Hızlı yaklaşık arama, kümeleme tabanlı
• IndexIVFPQ: Çok hızlı, düşük bellek, yaklaşık sonuçlar
• IndexHNSW: Hiyerarşik navigasyon, hızlı ve doğru

⚡ Performans İpuçları:
• GPU versiyonu çok daha hızlı (faiss-gpu)
• IVF için optimal küme sayısı: sqrt(n_vectors)
• PQ için boyut 8'in katı olmalı
• Büyük veri setleri için IVF + PQ kombinasyonu

🔧 Pratik Kullanım:
• Web araması: HNSW
• Öneri sistemleri: IVF
• Real-time arama: Flat
• Çok büyük veri: IVF + PQ
""")

# Memory usage analizi
index_flat = faiss.IndexFlatIP(dimension)
index_flat.add(vectors)

print(f"\n💾 Bellek Kullanımı:")
print(f"• Vektör verisi: {vectors.nbytes / 1024 / 1024:.1f} MB")
print(f"• Index boyutu: ~{vectors.nbytes / 1024 / 1024:.1f} MB (Flat)")
print(f"• Toplam: ~{2 * vectors.nbytes / 1024 / 1024:.1f} MB")

print("\n✅ FAISS öğreticisi tamamlandı!")
print("📁 Performance grafiği kaydedildi: faiss_performance.png")