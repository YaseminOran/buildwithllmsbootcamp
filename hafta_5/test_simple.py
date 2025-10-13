"""
Basit LangChain Test
"""

from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_openai import OpenAI
    print("✅ langchain_openai import başarılı")
except ImportError as e:
    print(f"❌ langchain_openai import hatası: {e}")

try:
    from langchain.prompts import PromptTemplate
    print("✅ langchain.prompts import başarılı")
except ImportError as e:
    print(f"❌ langchain.prompts import hatası: {e}")

try:
    from langchain.chains import LLMChain
    print("✅ langchain.chains import başarılı")
except ImportError as e:
    print(f"❌ langchain.chains import hatası: {e}")

# API key kontrolü
import os
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key != "your-openai-api-key-here":
    print("✅ OpenAI API key yüklendi")
    
    # Basit LLM testi
    try:
        llm = OpenAI(temperature=0.7, max_tokens=50)
        response = llm.invoke("Merhaba, nasılsın?")
        print(f"✅ LLM testi başarılı: {response}")
    except Exception as e:
        print(f"❌ LLM testi hatası: {e}")
else:
    print("❌ OpenAI API key bulunamadı veya varsayılan değer")
    print("   .env dosyasında OPENAI_API_KEY=your-actual-key yazın")