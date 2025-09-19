"""
Microsoft DialoGPT ile Konuşma Sistemi

ÖNEMLI NOTLAR:
1. DialoGPT conversation-focused bir model ama perfect değil
2. Bazen "I can't help you" gibi cevaplar verebilir
3. Bu normal - model training data'sına bağlı
4. Daha basit, direkt sorular daha iyi sonuç verir
5. Complex conversation context'lerden kaçınmak better

Problem: Conversation history'de eski Bot cevapları karışıyor
Çözüm: Sadece son kullanıcı sorusuna odaklanıp, sadece yeni Bot cevabını alırız
"""

import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline

def create_dialogpt_generator():
    """DialoGPT model pipeline'ını oluşturur"""
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")  # .env: HF_TOKEN=hf_...
    
    MODEL_ID = "microsoft/DialoGPT-medium"
    print(f"Model yükleniyor: {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    
    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=80,  # Daha uzun cevaplar için
        do_sample=True,
        temperature=0.8,    # Biraz daha yaratıcı
        top_p=0.9,          # Nucleus sampling
        repetition_penalty=1.1,  # Tekrar eden kelimelerden kaçın
        token=HF_TOKEN,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return generator

def get_bot_response(generator, conversation_prompt):
    """Konuşma prompt'ından sadece yeni Bot cevabını döndürür"""
    # Generate response
    response = generator(conversation_prompt)
    full_generated = response[0]["generated_text"]
    
    # Sadece yeni eklenen kısmı al (orijinal prompt'tan sonraki kısım)
    new_content = full_generated[len(conversation_prompt):].strip()
    
    # Eğer Bot: ile başlıyorsa onu kaldır, sadece cevabı al
    if new_content.startswith("Bot:"):
        new_content = new_content[4:].strip()
    
    # Eğer sonraki Human: kısmı varsa onu kes
    if "Human:" in new_content:
        new_content = new_content.split("Human:")[0].strip()
    
    return new_content

def main():
    """Ana demo fonksiyonu"""
    generator = create_dialogpt_generator()
    
    print("\n" + "="*60)
    print("MICROSOFT DIALOGPT CONVERSATION TEST")
    print("="*60)
    
    # Konuşma senaryoları - DialoGPT için optimize edilmiş
    conversations = [
        {
            "context": "Human: Tell me about language models. Bot:",
            "expected": "Direkt soru: Tell me about language models"
        },
        {
            "context": "Human: What is artificial intelligence? Bot:",
            "expected": "AI hakkında soru"
        },
        {
            "context": "Human: How does machine learning work? Bot:",
            "expected": "ML hakkında soru"
        },
        {
            "context": "Human: What is Python programming? Bot:",
            "expected": "Python hakkında soru"
        },
        {
            "context": "Human: Explain neural networks. Bot:",
            "expected": "Neural network açıklaması"
        }
    ]
    
    for i, conv in enumerate(conversations, 1):
        print(f"\n{i}. KONUŞMA:")
        print(f"Prompt Context: {conv['context']}")
        print(f"Beklenen: {conv['expected']}")
        print("-" * 50)
        
        try:
            bot_response = get_bot_response(generator, conv['context'])
            print(f"Bot Cevabı: {bot_response}")
        except Exception as e:
            print(f"HATA: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()