"""
Hafta 5 - Bölüm 3: Tools ve Agents
LangChain ile tool kullanımı ve agent oluşturma
"""

import os
import requests
import json
from datetime import datetime
from langchain_openai import OpenAI
from langchain.agents import Tool, AgentType, initialize_agent, create_react_agent
from langchain.tools import BaseTool
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# LLM'i başlat
llm = OpenAI(temperature=0, max_tokens=200, request_timeout=15)

# =============================================================================
# BASIT TOOL ÖRNEKLERİ
# =============================================================================

def get_current_time(query: str) -> str:
    """Şu anki zamanı döndürür"""
    now = datetime.now()
    return f"Şu anki tarih ve saat: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def simple_calculator(expression: str) -> str:
    """Basit matematik işlemleri yapar"""
    try:
        # Güvenlik için sadece belirli karakterlere izin ver
        allowed_chars = "0123456789+-*/.() "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Sonuç: {result}"
        else:
            return "Sadece temel matematik işlemleri desteklenir."
    except:
        return "Geçersiz matematik ifadesi."

def text_length_counter(text: str) -> str:
    """Metin uzunluğunu sayar"""
    word_count = len(text.split())
    char_count = len(text)
    return f"Kelime sayısı: {word_count}, Karakter sayısı: {char_count}"

def basic_tools_example():
    """Basit tool'ları kullanma örneği"""
    print("=" * 60)
    print("1. BASIT TOOLS KULLANIMI")
    print("=" * 60)
    
    # Tool'ları oluştur
    tools = [
        Tool(
            name="get_current_time",
            func=get_current_time,
            description="Şu anki tarih ve saati öğrenmek için kullanın"
        ),
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın. Örnek: 2+2 veya 10*5"
        ),
        Tool(
            name="text_counter",
            func=text_length_counter,
            description="Bir metnin kelime ve karakter sayısını bulmak için kullanın"
        )
    ]
    
    # Agent'ı oluştur
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations=3
    )
    
    # Test sorularını çalıştır
    questions = [
        "Şu anki saat kaç?",
        "25 çarpı 4 kaç eder?",
        "Bu metin kaç kelimeden oluşuyor: 'LangChain ile agent geliştiriyoruz'?"
    ]
    
    for question in questions:
        print(f"\nSoru: {question}")
        try:
            response = agent.run(question)
            print(f"Cevap: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return agent

# =============================================================================
# ÖZEL TOOL SINIFI
# =============================================================================

class WeatherToolInput(BaseModel):
    city: str = Field(description="Hava durumunu öğrenmek istediğiniz şehir adı")

class WeatherTool(BaseTool):
    name: str = "weather_tool"
    description: str = "Herhangi bir şehrin hava durumunu öğrenmek için kullanın"
    args_schema: Type[BaseModel] = WeatherToolInput

    def _run(
        self, 
        city: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Hava durumu bilgisi döndürür (simüle edilmiş)"""
        # Gerçek API yerine simüle edilmiş veri
        weather_data = {
            "istanbul": "İstanbul: 15°C, Parçalı bulutlu",
            "ankara": "Ankara: 12°C, Açık",
            "izmir": "İzmir: 18°C, Güneşli",
            "bursa": "Bursa: 14°C, Yağmurlu"
        }
        
        city_lower = city.lower()
        if city_lower in weather_data:
            return weather_data[city_lower]
        else:
            return f"{city} şehri için hava durumu bilgisi bulunamadı. Mevcut şehirler: İstanbul, Ankara, İzmir, Bursa"

    async def _arun(
        self,
        city: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async versiyon"""
        raise NotImplementedError("Bu tool async desteklemez")

class NewsToolInput(BaseModel):
    topic: str = Field(description="Haber konusu")

class NewsTool(BaseTool):
    name: str = "news_tool"
    description: str = "Belirli bir konu hakkında güncel haberleri getirir"
    args_schema: Type[BaseModel] = NewsToolInput
    
    def _run(
        self, 
        topic: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Simüle edilmiş haber verisi"""
        news_data = {
            "teknoloji": [
                "Yapay zeka alanında yeni gelişmeler",
                "Quantum bilgisayarlarda büyük adım",
                "5G teknolojisi yaygınlaşıyor"
            ],
            "spor": [
                "Futbol liginde heyecanlı maçlar",
                "Olimpiyat hazırlıkları devam ediyor",
                "Yeni spor kompleksi açıldı"
            ],
            "ekonomi": [
                "Borsa günü yükselişle kapattı",
                "Yeni yatırım teşvikleri açıklandı",
                "Dijital para birimlerinde hareket"
            ]
        }
        
        topic_lower = topic.lower()
        if topic_lower in news_data:
            news = news_data[topic_lower]
            return f"{topic} konusundaki güncel haberler:\n" + "\n".join([f"• {n}" for n in news])
        else:
            return f"{topic} konusunda haber bulunamadı. Mevcut konular: teknoloji, spor, ekonomi"

def custom_tools_example():
    """Özel tool sınıfları kullanma örneği"""
    print("\n" + "=" * 60)
    print("2. ÖZEL TOOL SINIFLARI")
    print("=" * 60)
    
    # Özel tool'ları oluştur
    tools = [
        WeatherTool(),
        NewsTool(),
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın"
        )
    ]
    
    # Agent'ı oluştur
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    
    # Test sorularını çalıştır
    questions = [
        "İstanbul'da hava nasıl?",
        "Teknoloji konusunda güncel haberler neler?",
        "50 bölü 5 kaç eder?"
    ]
    
    for question in questions:
        print(f"\nSoru: {question}")
        try:
            response = agent.run(question)
            print(f"Cevap: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return agent

# =============================================================================
# MEMORY İLE AGENT
# =============================================================================

def memory_agent_example():
    """Memory ile agent kullanımı"""
    print("\n" + "=" * 60)
    print("3. MEMORY İLE AGENT")
    print("=" * 60)
    
    # Memory oluştur
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Tool'lar
    tools = [
        Tool(
            name="get_time",
            func=get_current_time,
            description="Şu anki zamanı öğrenmek için kullanın"
        ),
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın"
        ),
        WeatherTool()
    ]
    
    # Memory'li agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )
    
    # Sıralı sorular (önceki cevapları hatırlayacak)
    conversation = [
        "Merhaba! İstanbul'da hava nasıl?",
        "Bu sıcaklığı Fahrenheit'a çevirir misin?",
        "İlk sorumda hangi şehri sormuştum?",
        "Şu anki saat kaç?"
    ]
    
    for question in conversation:
        print(f"\nKullanıcı: {question}")
        try:
            response = agent.run(question)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return agent

# =============================================================================
# REACTİVE AGENT
# =============================================================================

def create_reactive_agent_example():
    """Modern ReAct agent örneği"""
    print("\n" + "=" * 60)
    print("4. REACT AGENT YAPISI")
    print("=" * 60)
    
    # Tool'ları tanımla
    tools = [
        Tool(
            name="search",
            func=lambda query: f"'{query}' konusunda arama yapıldı. İlgili bilgiler bulundu.",
            description="İnternet araması yapmak için kullanın"
        ),
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın"
        ),
        WeatherTool()
    ]
    
    try:
        # ReAct prompt'unu yükle
        prompt = hub.pull("hwchase17/react")
        
        # Agent'ı oluştur
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True
        )
        
        # Test soruları
        complex_questions = [
            "İstanbul'da hava 15 derece. Bu sıcaklığı Fahrenheit'a çevir ve sonucu açıkla.",
            "Python programlama dili hakkında arama yap."
        ]
        
        for question in complex_questions:
            print(f"\nKarmaşık Soru: {question}")
            try:
                response = agent_executor.invoke({"input": question})
                print(f"Detaylı Cevap: {response['output']}")
            except Exception as e:
                print(f"Hata: {e}")
                
    except Exception as e:
        print(f"ReAct agent oluşturulurken hata: {e}")
        print("Basit agent ile devam ediliyor...")

# =============================================================================
# PERFORMANS ANALİZİ
# =============================================================================

def agent_performance_analysis():
    """Agent performansını analiz etme"""
    print("\n" + "=" * 60)
    print("5. AGENT PERFORMANS ANALİZİ")
    print("=" * 60)
    
    print("""
    AGENT TÜRLERİ VE KULLANIM ALANLARI:
    
    1. ZERO_SHOT_REACT_DESCRIPTION
       ✓ En basit agent türü
       ✓ Tool açıklamalarını kullanır
       ✗ Karmaşık görevlerde yetersiz
       
    2. CONVERSATIONAL_REACT_DESCRIPTION  
       ✓ Memory destekli
       ✓ Önceki konuşmaları hatırlar
       ✓ İnsan benzeri sohbet
       
    3. REACT_DOCSTORE
       ✓ Dokuman arama için optimize
       ✓ Bilgi tabanlı görevler
       
    4. SELF_ASK_WITH_SEARCH
       ✓ Karmaşık sorular için
       ✓ Alt sorulara böler
       
    PERFORMANS İPUÇLARI:
    - Tool açıklamalarını net yazın
    - Verbose=True ile debug yapın
    - Max_iterations ile sınır koyun
    - Error handling ekleyin
    """)

if __name__ == "__main__":
    print("LANGCHAIN TOOLS VE AGENTS ÖRNEKLERİ")
    print("Bu örneklerde tool ve agent kullanımını öğreneceksiniz.\n")
    
    try:
        # Tool ve agent örneklerini çalıştır
        basic_tools_example()
        custom_tools_example()
        memory_agent_example()
        create_reactive_agent_example()
        agent_performance_analysis()
        
        print("\n" + "=" * 60)
        print("TÜM TOOL VE AGENT ÖRNEKLERİ TAMAMLANDI!")
        print("Artık kendi tool'larınızı ve agent'larınızı oluşturabilirsiniz.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Genel hata: {e}")
        print("OpenAI API anahtarınızı ve bağımlılıkları kontrol edin!")