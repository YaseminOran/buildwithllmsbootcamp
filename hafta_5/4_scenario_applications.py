"""
Hafta 5 - Bölüm 4: Senaryo Bazlı Uygulamalar
Gerçek hayat senaryoları ile LangChain kullanımı
"""

import os
import json
from datetime import datetime, timedelta
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import BaseTool
from typing import Optional, Type, List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# LLM'i başlat
llm = OpenAI(temperature=0.7)

# =============================================================================
# SENARYO 1: MÜŞTERİ HİZMETLERİ BOT'U
# =============================================================================

class CustomerServiceBot:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Müşteri bilgileri (simüle edilmiş veri tabanı)
        self.customer_db = {
            "12345": {
                "name": "Ahmet Yılmaz",
                "email": "ahmet@email.com",
                "orders": ["ORD001", "ORD002"],
                "status": "Premium"
            },
            "67890": {
                "name": "Elif Kaya",
                "email": "elif@email.com", 
                "orders": ["ORD003"],
                "status": "Standard"
            }
        }
        
        # Sipariş bilgileri
        self.order_db = {
            "ORD001": {"product": "Laptop", "status": "Delivered", "date": "2024-01-15"},
            "ORD002": {"product": "Mouse", "status": "Shipped", "date": "2024-01-20"},
            "ORD003": {"product": "Keyboard", "status": "Processing", "date": "2024-01-18"}
        }
        
        self.setup_tools()
        self.setup_agent()
    
    def get_customer_info(self, customer_id: str) -> str:
        """Müşteri bilgilerini getir"""
        if customer_id in self.customer_db:
            customer = self.customer_db[customer_id]
            return f"Müşteri: {customer['name']}, Durum: {customer['status']}, Email: {customer['email']}"
        return "Müşteri bulunamadı."
    
    def get_order_status(self, order_id: str) -> str:
        """Sipariş durumunu kontrol et"""
        if order_id in self.order_db:
            order = self.order_db[order_id]
            return f"Sipariş {order_id}: {order['product']}, Durum: {order['status']}, Tarih: {order['date']}"
        return "Sipariş bulunamadı."
    
    def create_support_ticket(self, issue: str) -> str:
        """Destek bileti oluştur"""
        ticket_id = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return f"Destek biletiniz oluşturuldu. Bilet No: {ticket_id}. Konunuz: {issue}"
    
    def setup_tools(self):
        """Tool'ları oluştur"""
        self.tools = [
            Tool(
                name="get_customer_info",
                func=self.get_customer_info,
                description="Müşteri ID'si ile müşteri bilgilerini getirmek için kullanın"
            ),
            Tool(
                name="get_order_status", 
                func=self.get_order_status,
                description="Sipariş ID'si ile sipariş durumunu kontrol etmek için kullanın"
            ),
            Tool(
                name="create_support_ticket",
                func=self.create_support_ticket,
                description="Müşteri sorunu için destek bileti oluşturmak için kullanın"
            )
        ]
    
    def setup_agent(self):
        """Agent'ı oluştur"""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    def handle_customer_query(self, query: str) -> str:
        """Müşteri sorgusunu işle"""
        system_prompt = f"""
        Sen yardımsever bir müşteri hizmetleri temsilcisisin. 
        Müşterilere nazik ve profesyonel şekilde yardım et.
        
        Mevcut araçlar:
        - Müşteri bilgileri sorgulama
        - Sipariş durumu kontrolü  
        - Destek bileti oluşturma
        
        Müşteri sorusu: {query}
        """
        
        return self.agent.run(system_prompt)

def customer_service_scenario():
    """Müşteri hizmetleri senaryosu"""
    print("=" * 60)
    print("SENARYO 1: MÜŞTERİ HİZMETLERİ BOT'U")
    print("=" * 60)
    
    bot = CustomerServiceBot()
    
    # Test senaryoları
    scenarios = [
        "Merhaba, 12345 ID'li müşteri olarak hesap bilgilerimi öğrenebilir miyim?",
        "ORD001 numaralı siparişimin durumu nedir?",
        "Aldığım ürün bozuk geldi, ne yapabilirim?",
        "67890 müşteri ID'mle son siparişlerimi görebilir miyim?"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Test Senaryosu {i} ---")
        print(f"Müşteri: {scenario}")
        
        try:
            response = bot.handle_customer_query(scenario)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return bot

# =============================================================================
# SENARYO 2: İÇERİK OLUŞTURMA ASISTANI
# =============================================================================

class ContentCreationAssistant:
    def __init__(self):
        self.setup_chains()
    
    def setup_chains(self):
        """İçerik oluşturma chain'lerini kur"""
        
        # 1. Konu araştırması
        self.research_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Bu konu hakkında detaylı araştırma yapın: {topic}
            
            Şunları içeren bir araştırma raporu hazırlayın:
            - Ana konuların özeti
            - Hedef kitle analizi
            - Trend analizi
            - Anahtar kelimeler
            """
        )
        self.research_chain = LLMChain(
            llm=llm,
            prompt=self.research_prompt,
            output_key="research"
        )
        
        # 2. İçerik planı oluşturma
        self.planning_prompt = PromptTemplate(
            input_variables=["topic", "research"],
            template="""
            Konu: {topic}
            Araştırma: {research}
            
            Bu bilgilere dayanarak detaylı bir içerik planı oluşturun:
            - Ana başlıklar
            - Alt başlıklar  
            - İçerik akışı
            - Call-to-action önerileri
            """
        )
        self.planning_chain = LLMChain(
            llm=llm,
            prompt=self.planning_prompt,
            output_key="content_plan"
        )
        
        # 3. İçerik yazımı
        self.writing_prompt = PromptTemplate(
            input_variables=["topic", "research", "content_plan"],
            template="""
            Konu: {topic}
            Araştırma: {research}
            İçerik Planı: {content_plan}
            
            Bu plan doğrultusunda SEO-friendly, ilgi çekici ve bilgilendirici bir blog yazısı yazın.
            Yazı 800-1000 kelime olsun.
            """
        )
        self.writing_chain = LLMChain(
            llm=llm,
            prompt=self.writing_prompt,
            output_key="final_content"
        )
        
        # 4. Tüm chain'leri birleştir
        self.overall_chain = SequentialChain(
            chains=[self.research_chain, self.planning_chain, self.writing_chain],
            input_variables=["topic"],
            output_variables=["research", "content_plan", "final_content"]
        )
    
    def create_content(self, topic: str):
        """İçerik oluşturma süreci"""
        print(f"\n'{topic}' konusunda içerik oluşturuluyor...\n")
        
        result = self.overall_chain({"topic": topic})
        
        print("🔍 ARAŞTIRMA RAPORU:")
        print("-" * 40)
        print(result["research"])
        
        print("\n📋 İÇERİK PLANI:")
        print("-" * 40)
        print(result["content_plan"])
        
        print("\n✍️ FINAL İÇERİK:")
        print("-" * 40)
        print(result["final_content"])
        
        return result

def content_creation_scenario():
    """İçerik oluşturma senaryosu"""
    print("\n" + "=" * 60)
    print("SENARYO 2: İÇERİK OLUŞTURMA ASISTANI")
    print("=" * 60)
    
    assistant = ContentCreationAssistant()
    
    # Test konuları
    topics = [
        "Sürdürülebilir yaşam tarzı",
        "Uzaktan çalışmanın geleceği"
    ]
    
    for topic in topics:
        print(f"\n{'='*20} {topic.upper()} {'='*20}")
        try:
            assistant.create_content(topic)
        except Exception as e:
            print(f"İçerik oluşturma hatası: {e}")
    
    return assistant

# =============================================================================
# SENARYO 3: EĞİTİM PLANLAMA ASISTANI
# =============================================================================

class EducationPlannerBot:
    def __init__(self):
        self.courses_db = {
            "python": {"duration": "8 hafta", "level": "Başlangıç", "topics": ["Değişkenler", "Fonksiyonlar", "OOP"]},
            "javascript": {"duration": "10 hafta", "level": "Başlangıç", "topics": ["DOM", "ES6", "React"]},
            "machine_learning": {"duration": "12 hafta", "level": "İleri", "topics": ["Algoritma", "Neural Networks", "Deep Learning"]},
            "data_science": {"duration": "16 hafta", "level": "Orta", "topics": ["Pandas", "Visualization", "Statistics"]}
        }
        self.setup_chains()
    
    def get_course_info(self, course: str) -> str:
        """Kurs bilgilerini getir"""
        course_key = course.lower().replace(" ", "_")
        if course_key in self.courses_db:
            info = self.courses_db[course_key]
            return f"Kurs: {course}, Süre: {info['duration']}, Seviye: {info['level']}, Konular: {', '.join(info['topics'])}"
        return f"'{course}' kursu bulunamadı."
    
    def setup_chains(self):
        """Eğitim planlama chain'lerini kur"""
        
        # Seviye değerlendirme
        self.assessment_prompt = PromptTemplate(
            input_variables=["student_background", "goals"],
            template="""
            Öğrenci Geçmişi: {student_background}
            Hedefler: {goals}
            
            Bu bilgilere dayanarak öğrencinin seviyesini değerlendirin ve uygun başlangıç noktasını önerin.
            """
        )
        self.assessment_chain = LLMChain(
            llm=llm,
            prompt=self.assessment_prompt,
            output_key="assessment"
        )
        
        # Kişisel plan oluşturma
        self.planning_prompt = PromptTemplate(
            input_variables=["student_background", "goals", "assessment"],
            template="""
            Geçmiş: {student_background}
            Hedefler: {goals}
            Değerlendirme: {assessment}
            
            Kişiselleştirilmiş 12 haftalık öğrenim planı oluşturun:
            - Haftalık konular
            - Pratik projeler
            - Değerlendirme kriterleri
            - Kaynak önerileri
            """
        )
        self.planning_chain = LLMChain(
            llm=llm,
            prompt=self.planning_prompt,
            output_key="learning_plan"
        )
        
        # Motivasyon ve takip
        self.motivation_prompt = PromptTemplate(
            input_variables=["learning_plan"],
            template="""
            Öğrenim Planı: {learning_plan}
            
            Bu plan için motivasyon stratejileri ve ilerleme takip yöntemleri önerin:
            - Günlük rutinler
            - Milestone'lar
            - Ödül sistemi
            - Zorluk anlarında yapılacaklar
            """
        )
        self.motivation_chain = LLMChain(
            llm=llm,
            prompt=self.motivation_prompt,
            output_key="motivation_plan"
        )
        
        # Tüm chain'leri birleştir
        self.overall_chain = SequentialChain(
            chains=[self.assessment_chain, self.planning_chain, self.motivation_chain],
            input_variables=["student_background", "goals"],
            output_variables=["assessment", "learning_plan", "motivation_plan"]
        )
    
    def create_learning_plan(self, background: str, goals: str):
        """Öğrenim planı oluştur"""
        result = self.overall_chain({
            "student_background": background,
            "goals": goals
        })
        
        print("📊 SEVİYE DEĞERLENDİRMESİ:")
        print("-" * 40)
        print(result["assessment"])
        
        print("\n📚 KİŞİSEL ÖĞRENME PLANI:")
        print("-" * 40)  
        print(result["learning_plan"])
        
        print("\n💪 MOTİVASYON STRATEJİLERİ:")
        print("-" * 40)
        print(result["motivation_plan"])
        
        return result

def education_planning_scenario():
    """Eğitim planlama senaryosu"""
    print("\n" + "=" * 60)
    print("SENARYO 3: EĞİTİM PLANLAMA ASISTANI") 
    print("=" * 60)
    
    planner = EducationPlannerBot()
    
    # Test öğrenci profilleri
    students = [
        {
            "background": "Bilgisayar mühendisliği mezunu, 2 yıl web geliştirme deneyimi",
            "goals": "Veri bilimci olmak ve makine öğrenimi projelerinde çalışmak"
        },
        {
            "background": "İşletme mezunu, programlama deneyimi yok",
            "goals": "Mobil uygulama geliştirici olmak"
        }
    ]
    
    for i, student in enumerate(students, 1):
        print(f"\n{'='*15} ÖĞRENCİ {i} {'='*15}")
        try:
            planner.create_learning_plan(
                student["background"], 
                student["goals"]
            )
        except Exception as e:
            print(f"Plan oluşturma hatası: {e}")
    
    return planner

# =============================================================================
# ANA FONKSİYON
# =============================================================================

if __name__ == "__main__":
    print("LANGCHAIN SENARYO BAZLI UYGULAMALAR")
    print("Gerçek hayat senaryoları ile LangChain kullanımı\n")
    
    try:
        # Senaryoları çalıştır
        customer_service_scenario()
        content_creation_scenario()
        education_planning_scenario()
        
        print("\n" + "=" * 60)
        print("TÜM SENARYOLAR TAMAMLANDI!")
        print("Bu örnekleri kendi projelerinizde referans olarak kullanabilirsiniz.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Genel hata: {e}")
        print("OpenAI API anahtarınızı kontrol edin!")