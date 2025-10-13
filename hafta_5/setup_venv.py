"""
Virtual Environment Kurulum Script
Hafta 5 için gerekli paketleri otomatik yükler
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Komut çalıştır ve sonucu göster"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} başarılı!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} başarısız!")
        print(f"Hata: {e.stderr}")
        return False

def setup_virtual_environment():
    """Virtual environment kurmak ve paketleri yüklemek"""
    print("=" * 60)
    print("🚀 HAFTA 5 - VIRTUAL ENVIRONMENT KURULUMU")
    print("=" * 60)
    
    # İşletim sistemi kontrolü
    os_type = platform.system()
    print(f"📟 İşletim Sistemi: {os_type}")
    
    # Python versiyonu kontrolü
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python Versiyonu: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 veya üzeri gerekli!")
        return False
    
    # Virtual environment oluştur
    venv_name = "hafta5_env"
    
    if not run_command(f"python -m venv {venv_name}", "Virtual environment oluşturma"):
        return False
    
    # Aktivasyon komutları
    if os_type == "Windows":
        activate_cmd = f"{venv_name}\\Scripts\\activate"
        pip_cmd = f"{venv_name}\\Scripts\\pip"
    else:
        activate_cmd = f"source {venv_name}/bin/activate"
        pip_cmd = f"{venv_name}/bin/pip"
    
    print(f"\n📋 Virtual Environment Aktivasyon Komutu:")
    print(f"   {activate_cmd}")
    
    # Pip upgrade
    if not run_command(f"{pip_cmd} install --upgrade pip", "Pip güncelleme"):
        return False
    
    # Requirements yükle
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Paket yükleme"):
        return False
    
    # Kurulum doğrulama
    print("\n🔍 Kurulum Doğrulaması...")
    
    test_imports = [
        "langchain",
        "openai", 
        "dotenv",
        "tiktoken",
        "pydantic"
    ]
    
    for package in test_imports:
        try:
            result = subprocess.run(
                f"{pip_cmd} show {package}", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"  ✅ {package}")
            else:
                print(f"  ❌ {package}")
        except:
            print(f"  ❌ {package} - kontrol hatası")
    
    # .env dosyası kontrolü
    if not os.path.exists(".env"):
        print("\n📝 .env dosyası oluşturuluyor...")
        with open(".env", "w", encoding="utf-8") as f:
            f.write("# OpenAI API Anahtarınızı buraya yazın\n")
            f.write("OPENAI_API_KEY=your-openai-api-key-here\n")
        print("✅ .env dosyası oluşturuldu!")
        print("⚠️  .env dosyasına OpenAI API anahtarınızı ekleymeyi unutmayın!")
    else:
        print("✅ .env dosyası mevcut")
    
    print("\n" + "=" * 60)
    print("🎉 KURULUM TAMAMLANDI!")
    print("=" * 60)
    
    print(f"""
📚 ÖNEMLİ BİLGİLER:

1. Virtual Environment Aktivasyonu:
   {activate_cmd}

2. .env Dosyası:
   - .env dosyasını düzenleyin
   - OpenAI API anahtarınızı ekleyin
   - OPENAI_API_KEY=your-actual-api-key

3. Örnekleri Çalıştırma:
   - Virtual environment'ı aktive edin
   - python 1_chains_basic.py
   - python 2_memory_examples.py
   - python 3_tools_and_agents.py
   - python 4_scenario_applications.py
   - python 5_streaming_examples.py

4. Sorun Giderme:
   - API anahtarı hatası: .env dosyasını kontrol edin
   - Import hatası: requirements.txt'yi kontrol edin
   - Version hatası: Python 3.8+ kullanın
""")
    
    return True

def create_test_script():
    """Test scripti oluştur"""
    test_content = '''"""
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
    
    print("\\n🎉 Tüm testler başarılı!")
    return True

if __name__ == "__main__":
    test_installation()
'''
    
    with open("test_installation.py", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print("✅ test_installation.py oluşturuldu")

if __name__ == "__main__":
    try:
        setup_virtual_environment()
        create_test_script()
    except KeyboardInterrupt:
        print("\n\n❌ Kurulum iptal edildi!")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")