"""
OpenAI ChatCompletion API Detaylı Kullanım Örnekleri
Farklı parametreler, streaming, conversation management
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def basic_chat_completion():
    """Temel ChatCompletion kullanımı"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Python'da liste ve tuple arasındaki fark nedir?"}
        ]
    )
    
    return response.choices[0].message.content

def chat_with_system_message():
    """System message ile davranış belirleme"""
    
    messages = [
        {
            "role": "system", 
            "content": "Sen bir Python öğretmeni olan yardımcısın. Açıklamalarını basit ve kod örnekleriyle destekle."
        },
        {
            "role": "user", 
            "content": "Decorators nasıl çalışır?"
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def conversation_management():
    """Konuşma geçmişi yönetimi"""
    
    conversation = [
        {"role": "system", "content": "Sen yardımcı bir AI asistanısın."},
        {"role": "user", "content": "Merhaba! Python öğrenmeye yeni başladım."},
    ]
    
    # İlk yanıt
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    
    # AI yanıtını konuşmaya ekle
    conversation.append({
        "role": "assistant", 
        "content": response1.choices[0].message.content
    })
    
    # Kullanıcının ikinci sorusu
    conversation.append({ #extend
        "role": "user", 
        "content": "Hangi konulardan başlamalıyım?"
    })
    
    # İkinci yanıt (önceki konuşmayı hatırlayarak)
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    
    return {
        "first_response": response1.choices[0].message.content,
        "second_response": response2.choices[0].message.content,
        "full_conversation": conversation
    }

def different_temperature_examples():
    """Farklı temperature değerleri ile yaratıcılık kontrolü"""
    
    prompt = "Yapay zeka teknolojisinin geleceği hakkında 2 cümle yaz."
    
    results = {}
    
    # Düşük temperature (0.1) - deterministik
    response_low = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    results["low_temperature"] = response_low.choices[0].message.content
    
    # Orta temperature (0.7) - dengeli
    response_mid = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    results["mid_temperature"] = response_mid.choices[0].message.content
    
    # Yüksek temperature (1.2) - yaratıcı
    response_high = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.2
    )
    results["high_temperature"] = response_high.choices[0].message.content
    
    return results

def streaming_example():
    """Streaming response örneği"""
    
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Machine Learning konusunda kısa bir özet yaz."}
        ],
        stream=True, #cevap parça parça token token gelir
        max_tokens=200
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True) 
    
    print()  # Yeni satır
    return full_response

def multiple_choices_example():
    """Birden fazla yanıt seçeneği alma"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Motivasyonel bir söz söyle."}
        ],
        n=3,  # 3 farklı yanıt
        temperature=0.9
    )
    
    choices = []
    for i, choice in enumerate(response.choices):
        choices.append({
            "choice_number": i + 1,
            "content": choice.message.content
        })
    
    return choices

def token_usage_tracking():
    """Token kullanımını takip etme"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "E-ticaret sitesi için basit bir kullanıcı kayıt sistemi nasıl tasarlanır?"}
        ],
        max_tokens=250
    )
    
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }

def model_comparison():
    """Farklı modelleri karşılaştırma"""
    
    prompt = "JavaScript'te async/await nasıl kullanılır?"
    
    models = ["gpt-3.5-turbo", "gpt-4"]
    results = {}
    
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            results[model] = response.choices[0].message.content
        except Exception as e:
            results[model] = f"Hata: {str(e)}"
    
    return results

if __name__ == "__main__":
    print("=== OPENAI CHATCOMPLETION API ÖRNEKLERİ ===\n")
    
    # 1. Temel kullanım
    print("1. Temel ChatCompletion:")
    try:
        result = basic_chat_completion()
        print(f"Yanıt: {result}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 2. System message ile
    print("2. System Message ile:")
    try:
        result = chat_with_system_message()
        print(f"Yanıt: {result}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 3. Konuşma yönetimi
    print("3. Konuşma Geçmişi Yönetimi:")
    try:
        result = conversation_management()
        print(f"İlk Yanıt: {result['first_response']}")
        print(f"İkinci Yanıt: {result['second_response']}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 4. Temperature karşılaştırması
    print("4. Farklı Temperature Değerleri:")
    try:
        result = different_temperature_examples()
        for temp_type, content in result.items():
            print(f"{temp_type}: {content}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 5. Streaming
    print("5. Streaming Response:")
    try:
        print("Streaming yanıt:")
        result = streaming_example()
        print(f"\nTam yanıt: {result}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 6. Birden fazla seçenek
    print("6. Birden Fazla Yanıt Seçeneği:")
    try:
        result = multiple_choices_example()
        for choice in result:
            print(f"Seçenek {choice['choice_number']}: {choice['content']}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 7. Token kullanımı
    print("7. Token Kullanımı Takibi:")
    try:
        result = token_usage_tracking()
        print(f"Yanıt: {result['content']}")
        print(f"Token Kullanımı: {result['usage']}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 8. Model karşılaştırması
    print("8. Model Karşılaştırması:")
    try:
        result = model_comparison()
        for model, response in result.items():
            print(f"{model}: {response[:100]}...")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")