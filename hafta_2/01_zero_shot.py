"""
Zero-Shot Prompting Örneği
Zero-shot: Herhangi bir örnek vermeden doğrudan görev tanımı ile prompt yazma
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def zero_shot_classification():
    """Sıfır örnekle metin sınıflandırma"""
    
    prompt = """Bu yorumun duygusal tonunu belirle (pozitif, negatif, nötr):

Yorum: "Bu ürün gerçekten harika! Çok memnun kaldım, herkese tavsiye ederim."

Duygusal ton:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    
    return response.choices[0].message.content

def zero_shot_translation():
    """Sıfır örnekle çeviri"""
    
    prompt = """Aşağıdaki Türkçe metni İngilizceye çevir:

"Bugün hava çok güzel. Parkta yürüyüş yapmaya gidiyorum."

İngilizce çeviri:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    
    return response.choices[0].message.content

def zero_shot_summary():
    """Sıfır örnekle özet çıkarma"""
    
    text = """
    Yapay zeka teknolojisi son yıllarda büyük gelişmeler göstermiştir. Özellikle büyük dil modelleri, 
    doğal dil işleme görevlerinde insan seviyesinde performans gösterebilmektedir. Bu teknolojiler 
    çeviri, özet çıkarma, soru yanıtlama gibi birçok alanda kullanılmaktadır. Ancak bu gelişmeler 
    beraberinde etik endişeler ve iş gücü üzerindeki potansiyel etkiler gibi konuları da gündeme 
    getirmektedir. Gelecekte bu teknolojilerin daha da gelişeceği ve hayatımızın daha fazla alanında 
    yer alacağı öngörülmektedir.
    """
    
    prompt = f"""Aşağıdaki metni 2-3 cümleyle özetle:

{text}

Özet:"""

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
    print("=== ZERO-SHOT PROMPTING ÖRNEKLERİ ===\n")
    
    # Duygusal analiz
    print("1. Duygusal Analiz:")
    try:
        result = zero_shot_classification()
        print(f"Sonuç: {result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Çeviri
    print("2. Çeviri:")
    try:
        result = zero_shot_translation()
        print(f"Sonuç: {result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # Özet çıkarma
    print("3. Özet Çıkarma:")
    try:
        result = zero_shot_summary()
        print(f"Sonuç: {result}\n")
    except Exception as e:
        print(f"Hata: {e}\n")