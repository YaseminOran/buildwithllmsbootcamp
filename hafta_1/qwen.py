"""
Qwen 2.5 Model ile Temel Text Generation
Problem: text-generation pipeline tüm metni (prompt + response) döndürür
Çözüm: Promptu çıkararak sadece botun cevabını alırız
"""

from transformers import AutoTokenizer, pipeline

def create_qwen_generator():
    """Qwen model pipeline'ını oluşturur"""
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"Model yükleniyor: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    
    return generator

def get_bot_response(generator, prompt):
    """Promptu gönderir ve sadece botun cevabını döndürür"""
    response = generator(prompt)
    generated_text = response[0]["generated_text"]
    
    # Sadece botun cevabını al (promptu çıkar)
    bot_response = generated_text[len(prompt):].strip()
    return bot_response

def main():
    """Ana fonksiyon - çeşitli promptları test eder"""
    generator = create_qwen_generator()
    
    test_prompts = [
        "LLM nedir? Kısaca açıkla.",
        "Python programlama dilinin avantajları neler?",
        "Yapay zeka ve makine öğrenmesi arasındaki fark nedir?",
        "Transformer mimarisi nasıl çalışır?"
    ]
    
    print("\n" + "="*60)
    print("QWEN 2.5 MODEL TEST SONUÇLARI")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. PROMPT: {prompt}")
        print("-" * 50)
        
        try:
            bot_response = get_bot_response(generator, prompt)
            print(f"BOT CEVABI: {bot_response}")
        except Exception as e:
            print(f"HATA: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()