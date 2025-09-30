"""
Role-Based Prompting Örneği
Role-based: Modele belirli bir rol vererek daha spesifik ve tutarlı yanıtlar alma
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def marketing_expert_role():
    """Pazarlama uzmanı rolünde danışmanlık"""
    
    messages = [
        {
            "role": "system",
            "content": """Sen deneyimli bir dijital pazarlama uzmanısın. 10 yıllık tecrüben var. 
            E-ticaret, sosyal medya pazarlama ve marka stratejileri konularında uzmansın. 
            Pratik, uygulanabilir ve ölçülebilir öneriler verirsin."""
        },
        {
            "role": "user",
            "content": """Yeni açılan kahve dükkanım için sosyal medya stratejisi önerir misin? 
            Hedef kitlem 25-40 yaş arası, kahve seven profesyoneller."""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def technical_writer_role():
    """Teknik yazar rolünde dokümantasyon yazma"""
    
    messages = [
        {
            "role": "system",
            "content": """Sen profesyonel bir teknik yazarsın. Karmaşık teknik konuları 
            basit, anlaşılır dilde açıklama konusunda uzmansın. API dokümantasyonları, 
            kullanıcı kılavuzları ve teknik makaleler yazarsın."""
        },
        {
            "role": "user", 
            "content": """REST API'nin temel kavramlarını yeni başlayan geliştiriciler 
            için açıklar mısın? HTTP metodları, endpoint'ler ve response kodları dahil."""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=400,
        temperature=0.5
    )
    
    return response.choices[0].message.content

def financial_advisor_role():
    """Mali müşavir rolünde finansal analiz"""
    
    messages = [
        {
            "role": "system",
            "content": """Sen sertifikalı bir mali müşavirsin. Şirket finansmanı, 
            yatırım analizi ve risk yönetimi konularında 15 yıllık deneyimin var. 
            Türkiye'deki vergi mevzuatını ve finansal düzenlemeleri çok iyi bilirsin."""
        },
        {
            "role": "user",
            "content": """Startup'ım için yatırımcı bulmak istiyorum. Mali tablolarımda 
            hangi metriklere odaklanmalıyım ve nasıl sunmalıyım?"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=350,
        temperature=0.6
    )
    
    return response.choices[0].message.content

def teacher_role():
    """Öğretmen rolünde eğitim içeriği"""
    
    messages = [
        {
            "role": "system",
            "content": """Sen deneyimli bir matematik öğretmenisin. Öğrencilerin 
            seviyesine uygun açıklamalar yapar, örneklerle desteklersin. 
            Karmaşık konuları basit adımlarla öğretme konusunda çok başarılısın."""
        },
        {
            "role": "user",
            "content": """Lise öğrencilerine logaritma konusunu nasıl açıklayabilirim? 
            Günlük hayattan örneklerle anlatabiliir misin?"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=350,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def psychologist_role():
    """Psikolog rolünde danışmanlık"""
    
    messages = [
        {
            "role": "system",
            "content": """Sen lisanslı bir klinik psikologsun. Stres yönetimi, 
            iletişim becerileri ve kişisel gelişim konularında uzmansın. 
            Empati kurarak, destekleyici ve yapıcı tavsiyelerde bulunursun."""
        },
        {
            "role": "user",
            "content": """İş hayatında stresle başa çıkma konusunda zorlanıyorum. 
            Hangi teknikleri uygulayabilirim?"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.8
    )
    
    return response.choices[0].message.content

def chef_role():
    """Şef rolünde yemek tarifi"""
    
    messages = [
        {
            "role": "system",
            "content": """Sen Michelin yıldızlı restoran deneyimi olan profesyonel 
            bir şefsin. Hem geleneksel hem modern mutfak tekniklerini bilirsin. 
            Tariflerini detaylı ve uygulanabilir şekilde verirsin."""
        },
        {
            "role": "user",
            "content": """Evde kolayca yapabileceğim, misafirlere ikram edebileceğim 
            şık bir tatlı tarifi önerir misin?"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== ROLE-BASED PROMPTING ÖRNEKLERİ ===\n")
    
    roles = [
        ("Pazarlama Uzmanı", marketing_expert_role),
        ("Teknik Yazar", technical_writer_role),
        ("Mali Müşavir", financial_advisor_role),
        ("Matematik Öğretmeni", teacher_role),
        ("Psikolog", psychologist_role),
        ("Profesyonel Şef", chef_role)
    ]
    
    for i, (role_name, role_function) in enumerate(roles, 1):
        print(f"{i}. {role_name} Rolü:")
        try:
            result = role_function()
            print(f"Yanıt:\n{result}\n")
            print("-" * 60)
        except Exception as e:
            print(f"Hata: {e}\n")
            print("-" * 60)