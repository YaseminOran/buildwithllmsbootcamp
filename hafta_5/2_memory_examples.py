"""
Hafta 5 - Bölüm 2: Memory Kullanımı
LangChain ile farklı memory türlerini öğrenme
"""

import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory
)

from dotenv import load_dotenv
load_dotenv()

# LLM'i başlat (döngü önlemek için max_tokens düşük)
llm = OpenAI(temperature=0.5, max_tokens=50, request_timeout=10)

def buffer_memory_example():
    """ConversationBufferMemory - Tüm sohbeti hatırlar"""
    print("=" * 60)
    print("1. BUFFER MEMORY - Tüm konuşmayı hatırlar")
    print("=" * 60)
    
    # Buffer memory oluştur
    memory = ConversationBufferMemory()
    
    # Konversasyon chain'i oluştur
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False  # Sonsuz döngü önlemek için false
    )
    
    # Sohbet simülasyonu
    print("Sohbet başlıyor...")
    
    response1 = conversation.predict(input="Merhaba! Benim adım Ahmet.")
    print(f"AI: {response1}")
    
    response2 = conversation.predict(input="Sen benim adımı hatırlıyor musun?")
    print(f"AI: {response2}")
    
    response3 = conversation.predict(input="Peki ben hangi konularda ilgileniyorum?")
    print(f"AI: {response3}")
    
    # Memory içeriğini göster
    print("\nMemory içeriği:")
    print(memory.buffer)
    
    return memory

def window_memory_example():
    """ConversationBufferWindowMemory - Son N mesajı hatırlar"""
    print("\n" + "=" * 60)
    print("2. WINDOW MEMORY - Son 2 mesajı hatırlar")
    print("=" * 60)
    
    # Window memory (sadece son 2 etkileşimi tutar)
    memory = ConversationBufferWindowMemory(k=2)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Birden fazla mesaj
    messages = [
        "İlk mesaj: Merhaba!",
        "İkinci mesaj: Bugün hava nasıl?",
        "Üçüncü mesaj: Spor hakkında konuşalım.",
        "Dördüncü mesaj: İlk mesajımı hatırlıyor musun?"
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"\n--- Mesaj {i} ---")
        response = conversation.predict(input=msg)
        print(f"AI: {response}")
    
    print("\nSon memory içeriği (sadece son 2 etkileşim):")
    print(memory.buffer)
    
    return memory

def summary_memory_example():
    """ConversationSummaryMemory - Konuşmayı özetler"""
    print("\n" + "=" * 60)
    print("3. SUMMARY MEMORY - Konuşmayı özetler")
    print("=" * 60)
    
    # Summary memory
    memory = ConversationSummaryMemory(llm=llm)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Uzun sohbet simülasyonu
    long_conversation = [
        "Merhaba! Ben bir yazılım geliştiricisiyim.",
        "Python ve JavaScript kullanıyorum.",
        "Yapay zeka konusunda çok ilgiliyim.",
        "Son zamanlarda LangChain öğreniyorum.",
        "Peki benim hakkımda neler hatırlıyorsun?"
    ]
    
    for msg in long_conversation:
        response = conversation.predict(input=msg)
        print(f"AI: {response}\n")
    
    print("Memory özeti:")
    print(memory.buffer)
    
    return memory

def summary_buffer_memory_example():
    """ConversationSummaryBufferMemory - Hibrit yaklaşım"""
    print("\n" + "=" * 60)
    print("4. SUMMARY BUFFER MEMORY - Hibrit yaklaşım")
    print("=" * 60)
    
    # Summary buffer memory (max 100 token)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Mesajlar
    messages = [
        "Merhaba! Bugün makine öğrenimi hakkında konuşmak istiyorum.",
        "Özellikle neural network'ler ilgimi çekiyor.",
        "TensorFlow ve PyTorch arasındaki farkları merak ediyorum.",
        "Hangi projelerde hangi framework'ü kullanmalıyım?",
        "Peki başlangıçta nelerden bahsetmiştik?"
    ]
    
    for msg in messages:
        response = conversation.predict(input=msg)
        print(f"AI: {response}\n")
    
    print("Memory durumu:")
    print(f"Moving summary: {memory.moving_summary_buffer}")
    print(f"Chat memory: {memory.chat_memory.messages}")
    
    return memory

def token_buffer_memory_example():
    """ConversationTokenBufferMemory - Token limiti ile"""
    print("\n" + "=" * 60)
    print("5. TOKEN BUFFER MEMORY - Token limiti ile")
    print("=" * 60)
    
    # Token buffer memory (max 50 token)
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=50
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Kısa mesajlar
    short_messages = [
        "Merhaba!",
        "Nasılsın?",
        "Bugün güzel bir gün.",
        "Programlama öğreniyorum.",
        "İlk mesajımı hatırlıyor musun?"
    ]
    
    for msg in short_messages:
        response = conversation.predict(input=msg)
        print(f"AI: {response}\n")
    
    print("Token buffer içeriği:")
    print(memory.buffer)
    
    return memory

def custom_memory_with_chain():
    """Özel memory implementasyonu"""
    print("\n" + "=" * 60)
    print("6. ÖZEL MEMORY KULLANIMI - Kişiselleştirilmiş")
    print("=" * 60)
    
    # Özel memory ile prompt
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    template = """
    Sen yardımsever bir asistansın. Kullanıcının önceki mesajlarını hatırla.
    
    Önceki konuşma:
    {chat_history}
    
    Kullanıcı: {input}
    
    Asistan:
    """
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=template
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    
    # Etkileşimler
    interactions = [
        "Bana Python hakkında bilgi ver.",
        "Peki bu bilgileri nasıl uygulayabilirim?",
        "İlk sorumda ne sormuştum?"
    ]
    
    for interaction in interactions:
        response = chain.run(input=interaction)
        print(f"Kullanıcı: {interaction}")
        print(f"Asistan: {response}\n")
    
    return chain

def memory_comparison():
    """Farklı memory türlerinin karşılaştırması"""
    print("\n" + "=" * 60)
    print("7. MEMORY TÜRLERİ KARŞILAŞTIRMASI")
    print("=" * 60)
    
    print("""
    MEMORY TÜRLERİ:
    
    1. ConversationBufferMemory
       ✓ Tüm konuşmayı hatırlar
       ✗ Uzun konuşmalarda çok yer kaplar
       
    2. ConversationBufferWindowMemory
       ✓ Sabit boyutta memory kullanır
       ✗ Eski bilgileri kaybeder
       
    3. ConversationSummaryMemory
       ✓ Uzun konuşmaları özetler
       ✗ Detay kaybı olabilir
       
    4. ConversationSummaryBufferMemory
       ✓ En iyi özelliklerini birleştirir
       ✓ Esnek ve verimli
       
    5. ConversationTokenBufferMemory
       ✓ Token limiti ile tam kontrol
       ✓ Maliyet optimizasyonu
    """)

if __name__ == "__main__":
    print("LANGCHAIN MEMORY ÖRNEKLERİ")
    print("Bu örneklerde farklı memory türlerini öğreneceksiniz.\n")
    
    try:
        # Memory örneklerini çalıştır
        buffer_memory_example()
        window_memory_example()
        summary_memory_example()
        summary_buffer_memory_example()
        token_buffer_memory_example()
        custom_memory_with_chain()
        memory_comparison()
        
        print("\n" + "=" * 60)
        print("TÜM MEMORY ÖRNEKLERİ TAMAMLANDI!")
        print("Projelerinizde hangi memory türünü kullanacağınıza karar verebilirsiniz.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("OpenAI API anahtarınızı kontrol edin!")