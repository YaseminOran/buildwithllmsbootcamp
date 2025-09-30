"""
Function Calling ile Akıllı Chatbot Sistemi
06_function_calling.py'dan inherit ederek genişletilmiş özellikler
"""

import os
import json
import requests
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

# 06'daki FunctionCallingChatbot'u import et
import importlib.util
spec = importlib.util.spec_from_file_location("function_calling", "06_function_calling.py")
function_calling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(function_calling)
FunctionCallingChatbot = function_calling.FunctionCallingChatbot

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SmartChatbot(FunctionCallingChatbot):
    def __init__(self):
        # Parent'tan inherit et
        super().__init__()
        
        # System message'ı güncelle
        self.conversation_history[0]["content"] = """Sen çok yetenekli bir AI asistanısın. Kullanıcılara yardım etmek için 
        çeşitli araçları kullanabilirsin. Her zaman dostça, yardımcı ve profesyonel ol.
        Fonksiyon çağırımı yaptığında kullanıcıya ne yaptığını kısaca açıkla."""
        
        # Yeni fonksiyonları ekle (06'daki temel fonksiyonlara ek olarak)
        self.available_functions.update({
            "search_web": self.search_web,
            "set_reminder": self.set_reminder,
            "create_todo": self.create_todo,
            "get_random_fact": self.get_random_fact
        })
        
        # Ek veri yapıları
        self.todo_list = []
        self.reminders = []
    
    # Yeni fonksiyonlar (parent'ta olmayanlar)
    
    def search_web(self, query):
        """Web araması yapar (demo)"""
        search_results = {
            "python": "Python, güçlü ve popüler bir programlama dilidir. 1991'de Guido van Rossum tarafından geliştirildi.",
            "yapay zeka": "Yapay zeka, makinelerin insan benzeri düşünme yetenekleri göstermesidir. Machine learning ve deep learning alt dalları vardır.",
            "openai": "OpenAI, yapay zeka araştırmaları yapan şirkettir. ChatGPT ve GPT modelleri geliştirmiştir.",
            "javascript": "JavaScript, web geliştirme için kullanılan programlama dilidir. Hem frontend hem backend'de kullanılabilir."
        }
        
        query_lower = query.lower()
        for key in search_results:
            if key in query_lower:
                return {
                    "query": query,
                    "result": search_results[key],
                    "status": "success",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        return {
            "query": query,
            "result": f"'{query}' hakkında genel bilgiler bulunmaktadır. Daha spesifik arama yapmayı deneyin.",
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def set_reminder(self, message, time_minutes):
        """Hatırlatıcı ayarlar"""
        reminder_time = datetime.now() + timedelta(minutes=time_minutes)
        reminder = {
            "id": len(self.reminders) + 1,
            "message": message,
            "time": reminder_time.strftime("%Y-%m-%d %H:%M"),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.reminders.append(reminder)
        
        return {
            "status": "success",
            "message": f"Hatırlatıcı ayarlandı: '{message}' - {reminder['time']}'da",
            "reminder_id": reminder["id"],
            "total_reminders": len(self.reminders)
        }
    
    def create_todo(self, task, priority="medium"):
        """Yapılacaklar listesine görev ekler"""
        todo_item = {
            "id": len(self.todo_list) + 1,
            "task": task,
            "priority": priority,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "completed": False
        }
        self.todo_list.append(todo_item)
        
        return {
            "status": "success",
            "message": f"Görev eklendi: '{task}' (Öncelik: {priority})",
            "todo_id": todo_item["id"],
            "total_todos": len(self.todo_list)
        }
    
    def get_random_fact(self, category="general"):
        """Rastgele ilginç bilgi verir"""
        facts = {
            "science": [
                "Bal hiç bozulmaz. Antik Mısır mezarlarında hala yenilebilir bal bulunmuştur.",
                "Ahtapotların 3 kalbi ve mavi kanı vardır.",
                "Işık hızı saniyede yaklaşık 300,000 km'dir."
            ],
            "technology": [
                "İlk bilgisayar virüsü 1986'da 'Brain' adıyla Pakistan'da yazıldı.",
                "QWERTY klavye düzeni yazı makinelerinin sıkışmasını önlemek için tasarlandı.",
                "Internet'in ilk web sitesi 1991'de yayınlandı."
            ],
            "general": [
                "Penguentler monogamdır, hayat boyu aynı partneri tercih ederler.",
                "Bananaların teknik olarak meyve değil, ot olduğu kabul edilir.",
                "İnsan beyni günde yaklaşık 70,000 düşünce üretir."
            ]
        }
        
        import random
        category_facts = facts.get(category, facts["general"])
        selected_fact = random.choice(category_facts)
        
        return {
            "category": category,
            "fact": selected_fact,
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_function_definitions(self):
        """Fonksiyon tanımlarını parent'tan al ve yenilerini ekle"""
        parent_functions = super().get_function_definitions()
        
        new_functions = [
            {
                "name": "search_web",
                "description": "Web'de arama yapar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Arama sorgusu"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "set_reminder",
                "description": "Gelecek için hatırlatıcı ayarlar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Hatırlatıcı mesajı"},
                        "time_minutes": {"type": "integer", "description": "Kaç dakika sonra hatırlatılacak"}
                    },
                    "required": ["message", "time_minutes"]
                }
            },
            {
                "name": "create_todo",
                "description": "Yapılacaklar listesine yeni görev ekler",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Yapılacak görev"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Görev önceliği"}
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "get_random_fact",
                "description": "Rastgele ilginç bilgi verir",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["science", "technology", "general"], "description": "Bilgi kategorisi"}
                    }
                }
            }
        ]
        
        return parent_functions + new_functions
    
    def get_conversation_summary(self):
        """Konuşma özetini parent'tan al ve genişlet"""
        summary = super().get_conversation_summary()
        
        summary.update({
            "todo_count": len(self.todo_list),
            "reminder_count": len(self.reminders),
            "todos": self.todo_list,
            "reminders": self.reminders
        })
        
        return summary

def main():
    """Ana demo fonksiyonu"""
    print("🤖 Akıllı Chatbot'a Hoş Geldiniz!")
    print("06'daki temel fonksiyonlar + genişletilmiş özellikler:")
    print("• Alan hesaplama, hava durumu, döviz, zaman, e-posta (06'dan)")
    print("• Web araması, hatırlatıcı, yapılacaklar listesi, rastgele bilgi (yeni)\n")
    print("Çıkmak için 'quit' yazın.\n")
    
    chatbot = SmartChatbot()
    
    while True:
        try:
            user_input = input("Sen: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çık']:
                summary = chatbot.get_conversation_summary()
                print(f"\n📊 Konuşma Özeti:")
                print(f"Toplam mesaj: {summary['total_messages']}")
                print(f"Fonksiyon çağırımları: {summary['function_calls_made']}")
                print(f"Aktif görevler: {summary['todo_count']}")
                print(f"Hatırlatıcılar: {summary['reminder_count']}")
                print("Görüşmek üzere! 👋")
                break
            
            if not user_input:
                continue
            
            # Chatbot yanıtı al
            print("\nBot: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGörüşmek üzere! 👋")
            break
        except Exception as e:
            print(f"\nHata: {e}\n")

if __name__ == "__main__":
    # Demo kullanım
    print("=== AKILLI CHATBOT - 06'dan GENİŞLETİLMİŞ ===\n")
    
    # Otomatik demo
    demo_bot = SmartChatbot()
    
    demo_questions = [
        "Merhaba! Neler yapabilirsin?",
        "Yarıçapı 5 olan dairenin alanı nedir?",  # 06'dan
        "İstanbul'da hava nasıl?",  # 06'dan
        "Python hakkında arama yap",  # Yeni
        "30 dakika sonra toplantım olduğunu hatırlat",  # Yeni
        "Market alışverişi yapmayı yüksek öncelikli görev olarak ekle",  # Yeni
        "Bana ilginç bir teknoloji bilgisi ver",  # Yeni
        "100 USD kaç TL eder?",  # 06'dan
        "test@example.com geçerli bir e-posta mı?"  # 06'dan
    ]
    
    for question in demo_questions:
        print(f"🗣️ Kullanıcı: {question}")
        response = demo_bot.chat(question)
        print(f"🤖 Bot: {response}\n")
        print("-" * 70)
    
    print("\n🎯 İnteraktif moda geçmek için main() fonksiyonunu çalıştırın!")
    
    # İnteraktif mod için uncomment edin:
    main()