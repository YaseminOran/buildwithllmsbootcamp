"""
Few-Shot Prompting Örneği
Few-shot: Modelin öğrenmesi için birkaç örnek vererek prompt yazma
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def few_shot_classification():
    """Few-shot ile duygusal analiz"""
    
    prompt = """Aşağıdaki örnekleri incele ve son yorumun duygusal tonunu belirle:

Örnek 1:
Yorum: "Bu ürün berbat, hiç beğenmedim!"
Duygu: negatif

Örnek 2:
Yorum: "Oldukça iyi bir ürün, memnunum."
Duygu: pozitif

Örnek 3:
Yorum: "Normal bir ürün, ne iyi ne kötü."
Duygu: nötr

Şimdi bu yorumu sınıflandır:
Yorum: "Harika bir deneyimdi! Kesinlikle tekrar alırım."
Duygu:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    
    return response.choices[0].message.content

def few_shot_entity_extraction():
    """Few-shot ile varlık çıkarma"""
    
    prompt = """Metinlerden kişi isimlerini çıkar:

Örnek 1:
Metin: "Ahmet Yılmaz bugün Ankara'ya gitti."
Kişi: Ahmet Yılmaz

Örnek 2:
Metin: "Fatma Hanım ve Mehmet Bey toplantıya katıldı."
Kişi: Fatma Hanım, Mehmet Bey

Örnek 3:
Metin: "İstanbul'da güzel bir gün geçirdik."
Kişi: -

Şimdi bu metni analiz et:
Metin: "Prof. Dr. Ayşe Kaya ve Mühendis Ali Demir projeyi tamamladı."
Kişi:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    
    return response.choices[0].message.content

def few_shot_text_formatting():
    """Few-shot ile metin formatlama"""
    
    prompt = """Verilen cümleleri başlık formatına çevir:

Örnek 1:
Girdi: "yapay zeka ve gelecek"
Çıktı: "Yapay Zeka ve Gelecek"

Örnek 2:
Girdi: "makine öğrenmesi algoritmaları"
Çıktı: "Makine Öğrenmesi Algoritmaları"

Örnek 3:
Girdi: "derin öğrenme modelleri"
Çıktı: "Derin Öğrenme Modelleri"

Şimdi bu cümleyi formatla:
Girdi: "doğal dil işleme teknikleri"
Çıktı:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    
    return response.choices[0].message.content

def few_shot_qa_format():
    """Few-shot ile soru-cevap formatı"""
    
    prompt = """Verilen bilgilerden soru-cevap çiftleri oluştur:

Örnek 1:
Bilgi: "Python 1991 yılında Guido van Rossum tarafından geliştirildi."
Soru: Python ne zaman ve kim tarafından geliştirildi?
Cevap: Python 1991 yılında Guido van Rossum tarafından geliştirildi.

Örnek 2:
Bilgi: "JavaScript web tarayıcılarında çalışan bir programlama dilidir."
Soru: JavaScript nerede çalışır?
Cevap: JavaScript web tarayıcılarında çalışır.

Şimdi bu bilgiden soru-cevap oluştur:
Bilgi: "Machine Learning, verilerdeki kalıpları öğrenen algoritmalar kullanır."
Soru:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== FEW-SHOT PROMPTING ÖRNEKLERİ ===\n")
    
    # Duygusal analiz
    print("1. Few-Shot Duygusal Analiz:")
    try:
        result = few_shot_classification()
        print(f"Sonuç: {result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Varlık çıkarma
    print("2. Few-Shot Varlık Çıkarma:")
    try:
        result = few_shot_entity_extraction()
        print(f"Sonuç: {result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Metin formatlama
    print("3. Few-Shot Metin Formatlama:")
    try:
        result = few_shot_text_formatting()
        print(f"Sonuç: {result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Soru-cevap formatı
    print("4. Few-Shot Soru-Cevap:")
    try:
        result = few_shot_qa_format()
        print(f"Sonuç: {result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")