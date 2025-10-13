"""
Hafta 5 - Bölüm 5: Streaming Output ve Canlı Veri Akışı
LangChain ile streaming ve real-time uygulamalar
"""

import os
import time
import asyncio
from typing import Any, Dict, List, Optional
from langchain_openai import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# ÖZEL CALLBACK HANDLER'LAR
# =============================================================================

class CustomStreamingHandler(BaseCallbackHandler):
    """Özel streaming handler"""
    
    def __init__(self):
        self.tokens = []
        self.current_response = ""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """LLM başladığında çağrılır"""
        print("🤖 AI yanıt oluşturuyor...\n")
        print("📝 Canlı Yanıt: ", end="", flush=True)
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Her yeni token geldiğinde çağrılır"""
        print(token, end="", flush=True)
        self.tokens.append(token)
        self.current_response += token
        time.sleep(0.05)  # Typing efekti için
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """LLM bittiğinde çağrılır"""
        print("\n\n✅ Yanıt tamamlandı!")
        print(f"📊 Toplam token sayısı: {len(self.tokens)}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> Any:
        """Hata oluştuğunda çağrılır"""
        print(f"\n❌ Hata: {error}")

class ProgressHandler(BaseCallbackHandler):
    """İlerleme gösterici handler"""
    
    def __init__(self):
        self.step_count = 0
        self.steps = ["🔍 Analiz", "💭 Düşünme", "✍️ Yazma", "🎯 Tamamlama"]
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print("📈 İşlem Başlıyor:")
        self.show_progress()
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        if len(token.strip()) > 0 and self.step_count < len(self.steps) - 1:
            if token in ['.', '!', '?', '\n']:
                self.step_count += 1
                self.show_progress()
    
    def show_progress(self):
        """İlerleme çubuğunu göster"""
        progress = "["
        for i, step in enumerate(self.steps):
            if i <= self.step_count:
                progress += f"✅ {step} "
            else:
                progress += f"⏳ {step} "
        progress += "]"
        print(f"\r{progress}", end="", flush=True)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.step_count = len(self.steps) - 1
        self.show_progress()
        print("\n🎉 İşlem Tamamlandı!\n")

# =============================================================================
# STREAMING ÖRNEKLERİ
# =============================================================================

def basic_streaming_example():
    """Temel streaming örneği"""
    print("=" * 60)
    print("1. TEMEL STREAMING OUTPUT")
    print("=" * 60)
    
    # Streaming handler ile LLM
    llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    print("Normal yanıt (streaming yok):")
    normal_llm = OpenAI(temperature=0.7)
    normal_response = normal_llm("Python hakkında kısa bir açıklama yaz.")
    print(normal_response)
    
    print("\n" + "-" * 40)
    print("Streaming yanıt:")
    streaming_response = llm("Python hakkında kısa bir açıklama yaz.")
    print("\n")
    
    return streaming_response

def custom_streaming_example():
    """Özel streaming handler örneği"""
    print("\n" + "=" * 60)
    print("2. ÖZEL STREAMING HANDLER")
    print("=" * 60)
    
    # Özel handler ile LLM
    custom_handler = CustomStreamingHandler()
    llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[custom_handler]
    )
    
    # Uzun bir prompt ile test
    prompt = """
    Yapay zeka teknolojisinin gelecekte topluma etkilerini detaylı olarak açıkla.
    Pozitif ve negatif etkileri ayrı ayrı ele al.
    """
    
    response = llm(prompt)
    
    print(f"\n📋 Handler Bilgileri:")
    print(f"- Toplanan token sayısı: {len(custom_handler.tokens)}")
    print(f"- İlk 5 token: {custom_handler.tokens[:5]}")
    print(f"- Son 5 token: {custom_handler.tokens[-5:]}")
    
    return custom_handler

def progress_streaming_example():
    """İlerleme gösterici ile streaming"""
    print("\n" + "=" * 60)
    print("3. İLERLEME GÖSTERİCİLİ STREAMING")
    print("=" * 60)
    
    progress_handler = ProgressHandler()
    llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[progress_handler]
    )
    
    prompt = """
    Bir startup'ın başarılı olması için gerekli 5 temel unsuru açıkla.
    Her unsur için detaylı açıklama yap.
    """
    
    response = llm(prompt)
    print(response)
    
    return progress_handler

# =============================================================================
# STREAMING CHAIN ÖRNEKLERİ
# =============================================================================

def streaming_chain_example():
    """Chain ile streaming"""
    print("\n" + "=" * 60)
    print("4. STREAMING CHAIN KULLANIMI")
    print("=" * 60)
    
    # Streaming LLM
    streaming_llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[CustomStreamingHandler()]
    )
    
    # Prompt template
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="""
        Bu konu hakkında yaratıcı bir hikaye yaz: {topic}
        Hikaye en az 200 kelime olsun ve heyecanlı detaylar içersin.
        """
    )
    
    # Chain oluştur
    chain = LLMChain(
        llm=streaming_llm,
        prompt=prompt
    )
    
    # Chain'i çalıştır
    print("Hikaye konusu: 'Uzayda kaybolmuş bir robot'")
    result = chain.run("uzayda kaybolmuş bir robot")
    
    return result

# =============================================================================
# REAL-TIME SOHBET SİMÜLASYONU
# =============================================================================

class RealTimeChatBot:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.streaming_handler = CustomStreamingHandler()
        
        self.llm = OpenAI(
            temperature=0.8,
            streaming=True,
            callbacks=[self.streaming_handler]
        )
        
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"],
            template="""
            Sen arkadaş canlısı bir sohbet botusun. Kullanıcıyla doğal bir sohbet et.
            
            Önceki konuşma:
            {chat_history}
            
            Kullanıcı: {user_input}
            
            Bot:
            """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
    
    def chat(self, user_input: str):
        """Kullanıcı girişini işle"""
        print(f"\n👤 Kullanıcı: {user_input}")
        
        # Typing indicator
        print("⌨️  Bot yazıyor", end="", flush=True)
        for i in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\n")
        
        # Streaming response
        response = self.chain.run(user_input=user_input)
        return response

def realtime_chat_example():
    """Real-time sohbet örneği"""
    print("\n" + "=" * 60)
    print("5. REAL-TIME SOHBET BOT'U")
    print("=" * 60)
    
    chatbot = RealTimeChatBot()
    
    # Simüle edilmiş sohbet
    conversation = [
        "Merhaba! Nasılsın?",
        "Bugün hava çok güzel, sen ne yapıyorsun?",
        "Yapay zeka hakkında ne düşünüyorsun?",
        "Bana bir şaka anlatır mısın?"
    ]
    
    for message in conversation:
        try:
            response = chatbot.chat(message)
            time.sleep(1)  # Sohbet aralığı
        except Exception as e:
            print(f"Sohbet hatası: {e}")
            break
    
    return chatbot

# =============================================================================
# ASYNC STREAMING ÖRNEKLERİ
# =============================================================================

async def async_streaming_example():
    """Asynchronous streaming örneği"""
    print("\n" + "=" * 60)
    print("6. ASYNC STREAMING (Simüle Edilmiş)")
    print("=" * 60)
    
    # Async streaming simülasyonu
    responses = [
        "Python çok güçlü bir programlama dilidir.",
        "Web geliştirme, veri analizi, yapay zeka alanlarında kullanılır.",
        "Söz dizimi basit ve okunabilir olduğu için öğrenmesi kolaydır.",
        "Geniş kütüphane ekosistemi sayesinde hızlı geliştirme sağlar."
    ]
    
    print("🚀 Async streaming başlıyor...\n")
    
    for i, response in enumerate(responses, 1):
        print(f"📦 Chunk {i}: ", end="", flush=True)
        
        # Her karakteri ayrı ayrı yazdır
        for char in response:
            print(char, end="", flush=True)
            await asyncio.sleep(0.03)
        
        print()  # Yeni satır
        await asyncio.sleep(0.5)  # Chunk arası bekleme
    
    print("\n✅ Async streaming tamamlandı!")

# =============================================================================
# PERFORMANS KARŞILAŞTIRMASI
# =============================================================================

def streaming_performance_comparison():
    """Streaming vs Normal performans karşılaştırması"""
    print("\n" + "=" * 60)
    print("7. PERFORMANS KARŞILAŞTIRMASI")
    print("=" * 60)
    
    prompt = "Python programlama dilinin avantajlarını listele ve açıkla."
    
    # Normal LLM - zaman ölç
    print("⏱️  Normal LLM testi...")
    normal_llm = OpenAI(temperature=0.7)
    start_time = time.time()
    normal_response = normal_llm(prompt)
    normal_time = time.time() - start_time
    
    print(f"Normal LLM süresi: {normal_time:.2f} saniye")
    print(f"Normal yanıt uzunluğu: {len(normal_response)} karakter\n")
    
    # Streaming LLM - zaman ölç  
    print("⏱️  Streaming LLM testi...")
    streaming_llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    start_time = time.time()
    streaming_response = streaming_llm(prompt)
    streaming_time = time.time() - start_time
    
    print(f"\nStreaming LLM süresi: {streaming_time:.2f} saniye")
    print(f"Streaming yanıt uzunluğu: {len(streaming_response)} karakter")
    
    print(f"\n📊 Analiz:")
    print(f"- Süre farkı: {abs(normal_time - streaming_time):.2f} saniye")
    print(f"- Streaming kullanıcı deneyimi: Daha iyi (canlı feedback)")
    print(f"- Normal LLM: Daha hızlı işlem (tek seferde)")

# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    print("LANGCHAIN STREAMING VE CANLI VERİ AKIŞI ÖRNEKLERİ")
    print("Bu örneklerde streaming output ve real-time uygulamaları öğreneceksiniz.\n")
    
    try:
        # Streaming örneklerini çalıştır
        basic_streaming_example()
        custom_streaming_example()
        progress_streaming_example()
        streaming_chain_example()
        realtime_chat_example()
        
        # Async örneği çalıştır
        asyncio.run(async_streaming_example())
        
        streaming_performance_comparison()
        
        print("\n" + "=" * 60)
        print("TÜM STREAMING ÖRNEKLERİ TAMAMLANDI!")
        print("Artık kendi real-time uygulamalarınızı geliştirebilirsiniz.")
        print("=" * 60)
        
        # Pratik ipuçları
        print("\n🎯 STREAMING İPUÇLARI:")
        print("1. Uzun yanıtlar için streaming kullanın")
        print("2. Kullanıcı deneyimini iyileştirir")
        print("3. Custom handler'lar ile özelleştirin")
        print("4. Progress indicator'lar ekleyin")
        print("5. Error handling'i unutmayın")
        
    except Exception as e:
        print(f"Genel hata: {e}")
        print("OpenAI API anahtarınızı kontrol edin!")

if __name__ == "__main__":
    main()