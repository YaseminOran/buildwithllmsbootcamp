"""
Chain of Thought (CoT) Prompting Örneği
CoT: Modelin adım adım düşünmesini sağlayarak daha doğru sonuçlar elde etme
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cot_math_problem():
    """Chain of Thought ile matematik problemi"""
    
    prompt = """Bu matematik problemini adım adım çöz:

Problem: Bir mağazada 25 TL'lik bir ürün %20 indirim yapılıyor. Daha sonra indirimi ürün %15 KDV ekleniyor. Son fiyat ne kadar olur?

Adım adım çözüm:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0
    )
    
    return response.choices[0].message.content

def cot_logical_reasoning():
    """Chain of Thought ile mantıksal akıl yürütme"""
    
    prompt = """Bu mantık problemini adım adım analiz et:

Problem: 
- Ahmet, Mehmet'ten daha uzun
- Mehmet, Ali'den daha uzun  
- Can, Ahmet'ten daha kısa
- Ali, Can'dan daha uzun

Bu bilgilere göre en uzundan en kısaya doğru sıralama nasıl olur?

Adım adım analiz:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0
    )
    
    return response.choices[0].message.content

def cot_text_analysis():
    """Chain of Thought ile metin analizi"""
    
    text = """
    "Şirketimizin bu çeyrek satışları beklenenden %15 daha düşük oldu. Ancak maliyetlerimizi 
    %20 azaltmayı başardık. Yeni ürün lansmanımız için pazarlama bütçesini artırmayı planlıyoruz."
    """
    
    prompt = f"""Bu şirket raporunu analiz et ve genel durumu değerlendir:

Metin: {text}

Adım adım analiz:
1. Satış performansı: 
2. Maliyet durumu:
3. Gelecek planları:
4. Genel değerlendirme:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0
    )
    
    return response.choices[0].message.content

def cot_problem_solving():
    """Chain of Thought ile problem çözme"""
    
    prompt = """Bu senaryoyu analiz et ve çözüm öner:

Senaryo: Bir e-ticaret sitesinde müşteri şikayetleri artıyor. Ana sorunlar:
- Ürün teslimat süreleri uzun
- Müşteri hizmetleri yavaş yanıt veriyor
- Web sitesi ara sıra çöküyor
- İade süreci karmaşık

Bu problemleri çözmek için adım adım strateji geliştirelim:

1. Problemlerin öncelik sırası:
2. Her problem için çözüm önerileri:
3. Uygulama planı:
4. Başarı ölçütleri:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0
    )
    
    return response.choices[0].message.content

def cot_decision_making():
    """Chain of Thought ile karar verme"""
    
    prompt = """Bu karar verme sürecini adım adım analiz et:

Durum: Yazılım geliştiricisi olarak yeni bir projede teknoloji seçimi yapman gerekiyor.
Seçenekler:
A) React + Node.js (Bildiğin teknoloji, hızlı geliştirme)
B) Vue.js + Python (Öğrenmek istediğin, yeni fırsatlar)  
C) Angular + Java (Şirketin standart teknolojisi)

Karar verme süreci:
1. Kriterleri belirle:
2. Her seçeneği değerlendir:
3. Artı/eksileri karşılaştır:
4. En iyi seçimi yap ve nedenini açıkla:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== CHAIN OF THOUGHT PROMPTING ÖRNEKLERİ ===\n")
    
    # Matematik problemi
    print("1. CoT Matematik Problemi:")
    try:
        result = cot_math_problem()
        print(f"Sonuç:\n{result}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Mantıksal akıl yürütme
    print("2. CoT Mantıksal Akıl Yürütme:")
    try:
        result = cot_logical_reasoning()
        print(f"Sonuç:\n{result}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Metin analizi
    print("3. CoT Metin Analizi:")
    try:
        result = cot_text_analysis()
        print(f"Sonuç:\n{result}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Problem çözme
    print("4. CoT Problem Çözme:")
    try:
        result = cot_problem_solving()
        print(f"Sonuç:\n{result}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Karar verme
    print("5. CoT Karar Verme:")
    try:
        result = cot_decision_making()
        print(f"Sonuç:\n{result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")