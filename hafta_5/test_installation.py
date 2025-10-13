"""
Test Script - Kurulum Doğrulaması
"""
import os
from dotenv import load_dotenv

def test_installation():
    """Kurulum test et"""
    print("🧪 KURULUM TESTİ")
    print("=" * 40)
    
    # Environment test
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key and api_key != "your-openai-api-key-here":
        print("✅ OpenAI API Key yüklendi")
    else:
        print("❌ OpenAI API Key bulunamadı veya varsayılan değer")
        return False
    
    # Package imports test
    try:
        import langchain
        print(f"✅ LangChain {langchain.__version__}")
    except ImportError:
        print("❌ LangChain import edilemedi")
        return False
    
    try:
        import openai
        print(f"✅ OpenAI {openai.__version__}")
    except ImportError:
        print("❌ OpenAI import edilemedi")
        return False
    
    try:
        from langchain.llms import OpenAI
        llm = OpenAI(temperature=0)
        print("✅ LangChain OpenAI LLM oluşturuldu")
    except Exception as e:
        print(f"❌ LLM oluşturma hatası: {e}")
        return False
    
    print("\n🎉 Tüm testler başarılı!")
    return True

if __name__ == "__main__":
    test_installation()
