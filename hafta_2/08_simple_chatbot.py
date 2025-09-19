"""
Basit Function Calling Chatbot
Temel fonksiyon çağırımı ile kolay anlaşılır chatbot örneği
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimpleChatbot:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": "Sen yardımcı bir AI asistanısın. Hesap makinesi ve not alma fonksiyonlarını kullanabilirsin."
            }
        ]
        self.notes = []
        
        # Kullanılabilir fonksiyonlar
        self.available_functions = {
            "calculator": self.calculator,
            "save_note": self.save_note,
            "list_notes": self.list_notes
        }
    
    def calculator(self, operation, num1, num2):
        """Basit hesap makinesi"""
        try:
            if operation == "toplama":
                result = num1 + num2
            elif operation == "çıkarma":
                result = num1 - num2
            elif operation == "çarpma":
                result = num1 * num2
            elif operation == "bölme":
                if num2 == 0:
                    return {"error": "Sıfıra bölme hatası"}
                result = num1 / num2
            else:
                return {"error": "Geçersiz işlem"}
            
            return {
                "operation": operation,
                "num1": num1,
                "num2": num2,
                "result": result
            }
        except Exception as e:
            return {"error": str(e)}
    
    def save_note(self, title, content):
        """Not kaydetme"""
        note = {
            "id": len(self.notes) + 1,
            "title": title,
            "content": content,
            "timestamp": "şimdi"
        }
        self.notes.append(note)
        
        return {
            "message": f"Not kaydedildi: '{title}'",
            "note_id": note["id"],
            "total_notes": len(self.notes)
        }
    
    def list_notes(self):
        """Notları listeler"""
        if not self.notes:
            return {
                "message": "Henüz kaydedilmiş not bulunmuyor.",
                "notes": [],
                "total_count": 0
            }
        
        return {
            "message": f"Toplam {len(self.notes)} not bulundu:",
            "notes": self.notes,
            "total_count": len(self.notes)
        }
    
    def get_functions(self):
        """Fonksiyon tanımları"""
        return [
            {
                "name": "calculator",
                "description": "Temel matematik işlemleri yapar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["toplama", "çıkarma", "çarpma", "bölme"],
                            "description": "Yapılacak işlem"
                        },
                        "num1": {
                            "type": "number",
                            "description": "İlk sayı"
                        },
                        "num2": {
                            "type": "number", 
                            "description": "İkinci sayı"
                        }
                    },
                    "required": ["operation", "num1", "num2"]
                }
            },
            {
                "name": "save_note",
                "description": "Not kaydeder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Not başlığı"
                        },
                        "content": {
                            "type": "string",
                            "description": "Not içeriği"
                        }
                    },
                    "required": ["title", "content"]
                }
            },
            {
                "name": "list_notes",
                "description": "Kayıtlı tüm notları listeler",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def chat(self, user_message):
        """Chat fonksiyonu"""
        # Kullanıcı mesajını ekle
        self.messages.append({"role": "user", "content": user_message})
        
        try:
            # OpenAI API çağrısı
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                functions=self.get_functions(),
                function_call="auto"
            )
            
            message = response.choices[0].message
            
            # Function call kontrolü
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                # Fonksiyon çağırma
                if function_name in self.available_functions:
                    result = self.available_functions[function_name](**function_args)
                else:
                    result = {"error": "Bilinmeyen fonksiyon"}
                
                # Fonksiyon sonucunu conversation'a ekle
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": message.function_call
                })
                
                self.messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
                
                # Final yanıt
                final_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=self.messages
                )
                
                final_answer = final_response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": final_answer})
                
                return final_answer
            
            else:
                # Normal yanıt
                answer = message.content
                self.messages.append({"role": "assistant", "content": answer})
                return answer
                
        except Exception as e:
            return f"Hata: {str(e)}"

def demo():
    """Demo kullanım"""
    print("🤖 Basit Chatbot Demo")
    print("Hesap makinesi ve not alma özelliklerim var!\n")
    
    bot = SimpleChatbot()
    
    test_messages = [
        "Merhaba!",
        "25 ile 17'yi topla",
        "120'yi 8'e böl", 
        "Bugün market listesi: süt, ekmek, yumurta - bu notu 'market' başlığıyla kaydet",
        "45 çarpı 3 kaç eder?",
        "Proje toplantısı: Yarın saat 14:00'da ofiste - bunu 'toplantı' başlığıyla kaydet",
        "Notlarımı göster",
        "Kaç notum var?"
    ]
    
    for msg in test_messages:
        print(f"👤 Sen: {msg}")
        response = bot.chat(msg)
        print(f"🤖 Bot: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    demo()