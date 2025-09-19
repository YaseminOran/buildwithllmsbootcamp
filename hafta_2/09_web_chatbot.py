"""
Web Arayüzlü Function Calling Chatbot
Flask ile basit web arayüzü
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global chatbot instance
chatbot_instance = None

class WebChatbot:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": "Sen web arayüzlü bir AI asistanısın. Kullanıcılara hesaplama, bilgi arama ve not alma konularında yardım ediyorsun."
            }
        ]
        self.user_data = {
            "notes": [],
            "calculations": []
        }
    
    def calculate(self, expression):
        """Matematik hesaplama"""
        try:
            # Basit güvenlik kontrolü
            allowed_chars = "0123456789+-*/()."
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return {"error": "Geçersiz karakter"}
            
            result = eval(expression)
            calculation = {
                "expression": expression,
                "result": result,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.user_data["calculations"].append(calculation)
            
            return calculation
        except Exception as e:
            return {"error": f"Hesaplama hatası: {str(e)}"}
    
    def save_note(self, note_text):
        """Not kaydetme"""
        note = {
            "id": len(self.user_data["notes"]) + 1,
            "text": note_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.user_data["notes"].append(note)
        
        return {
            "message": "Not kaydedildi",
            "note_id": note["id"],
            "total_notes": len(self.user_data["notes"])
        }
    
    def get_real_exchange_rate(self, from_currency, to_currency):
        """Gerçek zamanlı döviz kuru alır"""
        try:
            # Exchangerate-API (ücretsiz, API key gerektirmez)
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                rates = data.get('rates', {})
                
                if to_currency.upper() in rates:
                    return {
                        "rate": rates[to_currency.upper()],
                        "source": "real-time",
                        "success": True
                    }
            
            return {"success": False, "error": "API hatası"}
            
        except Exception as e:
            # Hata durumunda demo kurlara geri dön
            return {"success": False, "error": str(e)}
    
    def convert_currency_real(self, amount, from_currency, to_currency):
        """Gerçek zamanlı para birimi dönüştürme"""
        # Önce gerçek kur al
        rate_data = self.get_real_exchange_rate(from_currency, to_currency)
        
        if rate_data["success"]:
            rate = rate_data["rate"]
            converted = round(amount * rate, 2)
            return {
                "original_amount": amount,
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "converted_amount": converted,
                "exchange_rate": rate,
                "source": "real-time",
                "status": "success"
            }
        else:
            # Gerçek kur alınamazsa demo kurlara geri dön
            demo_rates = {
                "USD": {"TRY": 27.5, "EUR": 0.92, "GBP": 0.79},
                "TRY": {"USD": 0.036, "EUR": 0.033, "GBP": 0.029},
                "EUR": {"USD": 1.08, "TRY": 30.0, "GBP": 0.86},
                "GBP": {"USD": 1.26, "TRY": 34.5, "EUR": 1.16}
            }
            
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()
            
            if from_currency in demo_rates and to_currency in demo_rates[from_currency]:
                rate = demo_rates[from_currency][to_currency]
                converted = round(amount * rate, 2)
                return {
                    "original_amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "converted_amount": converted,
                    "exchange_rate": rate,
                    "source": "demo",
                    "status": "success",
                    "note": "Gerçek kur alınamadı, demo kur kullanıldı"
                }
            else:
                return {"status": "error", "message": "Desteklenmeyen döviz çifti"}

    def search_info(self, query):
        """Bilgi arama (demo)"""
        info_db = {
            "python": "Python, 1991'de Guido van Rossum tarafından geliştirilen yüksek seviyeli programlama dilidir.",
            "openai": "OpenAI, yapay zeka araştırmaları yapan şirkettir. ChatGPT ve GPT modelleri geliştirmiştir.",
            "javascript": "JavaScript, web geliştirme için kullanılan dinamik programlama dilidir.",
            "react": "React, Facebook tarafından geliştirilen kullanıcı arayüzü kütüphanesidir."
        }
        
        query_lower = query.lower()
        for key, value in info_db.items():
            if key in query_lower:
                return {"query": query, "result": value}
        
        return {"query": query, "result": f"'{query}' hakkında bilgi bulunamadı."}
    
    def list_notes(self):
        """Kayıtlı notları listeler"""
        if not self.user_data["notes"]:
            return {
                "message": "Henüz kaydedilmiş not bulunmuyor.",
                "notes": [],
                "total_count": 0
            }
        
        return {
            "message": f"Toplam {len(self.user_data['notes'])} not bulundu:",
            "notes": self.user_data["notes"],
            "total_count": len(self.user_data["notes"])
        }
    
    def get_functions(self):
        """Fonksiyon tanımları"""
        return [
            {
                "name": "calculate",
                "description": "Matematik hesaplaması yapar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Hesaplanacak matematik ifadesi"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "save_note",
                "description": "Not kaydeder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note_text": {
                            "type": "string",
                            "description": "Kaydedilecek not"
                        }
                    },
                    "required": ["note_text"]
                }
            },
            {
                "name": "convert_currency_real",
                "description": "Gerçek zamanlı döviz kurları ile para birimi dönüştürür",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Dönüştürülecek miktar"
                        },
                        "from_currency": {
                            "type": "string",
                            "description": "Kaynak para birimi (USD, EUR, TRY, GBP)"
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "Hedef para birimi (USD, EUR, TRY, GBP)"
                        }
                    },
                    "required": ["amount", "from_currency", "to_currency"]
                }
            },
            {
                "name": "search_info",
                "description": "Konu hakkında bilgi arar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Aranacak konu"
                        }
                    },
                    "required": ["query"]
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
        """Chat işlemi"""
        self.messages.append({"role": "user", "content": user_message})
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                functions=self.get_functions(),
                function_call="auto"
            )
            
            message = response.choices[0].message
            
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                # Fonksiyon çağırma
                if function_name == "calculate":
                    result = self.calculate(**function_args)
                elif function_name == "save_note":
                    result = self.save_note(**function_args)
                elif function_name == "convert_currency_real":
                    result = self.convert_currency_real(**function_args)
                elif function_name == "search_info":
                    result = self.search_info(**function_args)
                elif function_name == "list_notes":
                    result = self.list_notes()
                
                # Conversation'a ekle
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
                answer = message.content
                self.messages.append({"role": "assistant", "content": answer})
                return answer
                
        except Exception as e:
            return f"Hata: {str(e)}"

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chatbot</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chat-box { height: 400px; border: 1px solid #ddd; padding: 15px; overflow-y: auto; margin-bottom: 10px; background: #fafafa; border-radius: 5px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #007bff; color: white; text-align: right; }
        .bot { background: #e9ecef; color: #333; }
        .input-area { display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .input-area button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .input-area button:hover { background: #0056b3; }
        .stats { display: flex; gap: 20px; margin-top: 20px; }
        .stat-box { background: #e9ecef; padding: 10px; border-radius: 5px; flex: 1; text-align: center; }
        h1 { color: #333; text-align: center; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Chatbot with Function Calling</h1>
        <div id="chat-box" class="chat-box"></div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Mesajınızı yazın..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Gönder</button>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <strong>Hesaplamalar</strong><br>
                <span id="calc-count">0</span>
            </div>
            <div class="stat-box">
                <strong>Notlar</strong><br>
                <span id="note-count">0</span>
            </div>
            <div class="stat-box">
                <strong>Mesajlar</strong><br>
                <span id="message-count">0</span>
            </div>
        </div>
    </div>

    <script>
        let messageCount = 0;
        
        function addMessage(content, isUser) {
            const chatBox = document.getElementById('chat-box');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
            messageDiv.textContent = content;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            
            if (isUser) {
                messageCount++;
                document.getElementById('message-count').textContent = messageCount;
            }
        }
        
        function addLoadingMessage() {
            const chatBox = document.getElementById('chat-box');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot loading';
            messageDiv.id = 'loading-message';
            messageDiv.textContent = 'Bot düşünüyor...';
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function removeLoadingMessage() {
            const loadingMsg = document.getElementById('loading-message');
            if (loadingMsg) {
                loadingMsg.remove();
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            addLoadingMessage();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                const data = await response.json();
                removeLoadingMessage();
                addMessage(data.response, false);
                
                // İstatistikleri güncelle
                if (data.stats) {
                    document.getElementById('calc-count').textContent = data.stats.calculations;
                    document.getElementById('note-count').textContent = data.stats.notes;
                }
                
            } catch (error) {
                removeLoadingMessage();
                addMessage('Bağlantı hatası: ' + error.message, false);
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Başlangıç mesajı
        addMessage('Merhaba! Size nasıl yardımcı olabilirim? Hesaplama yapabilirim, not alabilir ve bilgi arayabilirim.', false);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/chat', methods=['POST'])
def chat():
    global chatbot_instance
    
    if chatbot_instance is None:
        chatbot_instance = WebChatbot()
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Mesaj boş olamaz'})
    
    try:
        response = chatbot_instance.chat(user_message)
        
        stats = {
            'calculations': len(chatbot_instance.user_data['calculations']),
            'notes': len(chatbot_instance.user_data['notes'])
        }
        
        return jsonify({
            'response': response,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🌐 Web Chatbot başlatılıyor...")
    print("Tarayıcınızda http://localhost:5000 adresini açın")
    
    # Flask'ı pip ile yüklemek gerekebilir: pip install flask
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Flask başlatılamadı: {e}")
        print("Flask'ı yüklemek için: pip install flask")