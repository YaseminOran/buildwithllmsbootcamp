@echo off
REM Build with LLMs Bootcamp - Windows Kurulum ve Başlatma Scripti (ensurepip Fix)
REM Bu script ensurepip sorunu için alternatif çözüm sağlar

setlocal enabledelayedexpansion

REM Renkli output için Windows
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Logo ve başlık
echo %BLUE%
echo =================================================
echo    🚀 BUILD WITH LLMS BOOTCAMP KURULUM (WINDOWS FIX)
echo =================================================
echo %NC%

REM Gereksinimler kontrolü
echo %YELLOW%📋 Sistem gereksinimleri kontrol ediliyor...%NC%

REM Python versiyonu kontrolü
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Python bulunamadı. Lütfen Python 3.8+ yükleyin.%NC%
    echo Python'u https://python.org adresinden indirin
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%✅ Python !PYTHON_VERSION! bulundu%NC%

REM Sanal ortam oluşturma (ensurepip olmadan)
set "VENV_NAME=llm_bootcamp_env"
echo %YELLOW%🔧 Sanal ortam oluşturuluyor (pip olmadan): !VENV_NAME!%NC%

if exist "!VENV_NAME!" (
    echo %YELLOW%⚠️  Sanal ortam zaten mevcut. Yeniden oluşturuluyor...%NC%
    rmdir /s /q "!VENV_NAME!"
)

REM Sanal ortamı pip olmadan oluştur
python -m venv "!VENV_NAME!" --without-pip
if errorlevel 1 (
    echo %RED%❌ Sanal ortam oluşturulamadı%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Sanal ortam oluşturuldu (pip olmadan)%NC%

REM Sanal ortamı aktive et
echo %YELLOW%🔄 Sanal ortam aktive ediliyor...%NC%
call "!VENV_NAME!\Scripts\activate.bat"

REM pip'i manuel olarak yükle
echo %YELLOW%📥 pip manuel olarak yükleniyor...%NC%

REM get-pip.py indir ve yükle
powershell -Command "Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py"
if not exist get-pip.py (
    echo %RED%❌ get-pip.py indirilemedi%NC%
    echo %YELLOW%Manuel olarak https://bootstrap.pypa.io/get-pip.py adresinden indirin%NC%
    pause
    exit /b 1
)

python get-pip.py
if errorlevel 1 (
    echo %RED%❌ pip yüklenemedi%NC%
    pause
    exit /b 1
)

echo %GREEN%✅ pip başarıyla yüklendi%NC%

REM get-pip.py dosyasını temizle
del get-pip.py

REM pip, setuptools ve wheel'i güncelle
echo %YELLOW%⬆️  pip, setuptools ve wheel güncelleniyor...%NC%
python -m pip install --upgrade pip setuptools wheel

REM GPU desteği kontrolü
echo %YELLOW%🔍 GPU desteği kontrol ediliyor...%NC%

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠️  NVIDIA GPU bulunamadı - CPU modunda çalışacak%NC%
    set "GPU_TYPE=cpu"
) else (
    echo %GREEN%✅ NVIDIA GPU bulundu%NC%
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | findstr /n "^"
    set "GPU_TYPE=cuda"
)

REM PyTorch yükleme - GPU tipine göre
echo %YELLOW%🔥 PyTorch yükleniyor...%NC%

if "!GPU_TYPE!"=="cuda" (
    echo %BLUE%CUDA desteği ile PyTorch yükleniyor...%NC%
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo %BLUE%CPU versiyonu PyTorch yükleniyor...%NC%
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo %RED%❌ PyTorch yüklenemedi%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ PyTorch yüklendi%NC%

REM requirements.txt'den diğer paketleri yükle
echo %YELLOW%📦 Diğer bağımlılıklar yükleniyor...%NC%

if "!GPU_TYPE!"=="cpu" (
    REM CPU için bazı GPU-specific paketleri çıkar
    findstr /v "bitsandbytes GPUtil" requirements.txt > requirements_cpu.txt
    pip install -r requirements_cpu.txt
    del requirements_cpu.txt
) else (
    pip install -r requirements.txt
)

if errorlevel 1 (
    echo %YELLOW%⚠️  Bazı paketler yüklenemedi, devam ediliyor...%NC%
) else (
    echo %GREEN%✅ Tüm bağımlılıklar yüklendi%NC%
)

REM Spacy modeli yükleme
echo %YELLOW%🔤 Spacy dil modeli yükleniyor...%NC%
python -m spacy download en_core_web_sm

REM NLTK verileri yükleme
echo %YELLOW%📚 NLTK verileri yükleniyor...%NC%
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True); print('NLTK verileri yüklendi')"

REM Kurulum testi
echo %YELLOW%🧪 Kurulum test ediliyor...%NC%

python -c "import torch; import transformers; import numpy as np; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ Transformers: {transformers.__version__}'); print(f'✅ NumPy: {np.__version__}'); print('✅ CUDA destekleniyor' if torch.cuda.is_available() else '✅ CPU modunda çalışıyor')"

REM Hızlı test scripti oluştur (sadece yoksa)
if not exist quick_test.py (
    echo %YELLOW%📝 Test scripti oluşturuluyor...%NC%
    
    echo #!/usr/bin/env python3 > quick_test.py
    echo """ >> quick_test.py
    echo Hızlı kurulum testi >> quick_test.py
    echo """ >> quick_test.py
    echo. >> quick_test.py
    echo import torch >> quick_test.py
    echo from transformers import pipeline >> quick_test.py
    echo import time >> quick_test.py
    echo. >> quick_test.py
    echo def test_installation(): >> quick_test.py
    echo     print("🧪 Kurulum testi başlatılıyor...") >> quick_test.py
    echo. >> quick_test.py
    echo     # Device kontrolü >> quick_test.py
    echo     if torch.cuda.is_available(): >> quick_test.py
    echo         device = "cuda" >> quick_test.py
    echo         print(f"✅ CUDA GPU: {torch.cuda.get_device_name(0)}") >> quick_test.py
    echo     else: >> quick_test.py
    echo         device = "cpu" >> quick_test.py
    echo         print("✅ CPU modunda çalışıyor") >> quick_test.py
    echo. >> quick_test.py
    echo     # Basit sentiment analysis testi >> quick_test.py
    echo     print("\n🔍 Sentiment analysis testi...") >> quick_test.py
    echo     classifier = pipeline("sentiment-analysis") >> quick_test.py
    echo. >> quick_test.py
    echo     start_time = time.time() >> quick_test.py
    echo     result = classifier("This is a great bootcamp!") >> quick_test.py
    echo     end_time = time.time() >> quick_test.py
    echo. >> quick_test.py
    echo     print(f"Sonuç: {result[0]['label']} (güven: {result[0]['score']:.4f})") >> quick_test.py
    echo     print(f"Süre: {end_time - start_time:.4f} saniye") >> quick_test.py
    echo. >> quick_test.py
    echo     print("\n🎉 Test başarılı! Sistem kullanıma hazır.") >> quick_test.py
    echo. >> quick_test.py
    echo if __name__ == "__main__": >> quick_test.py
    echo     test_installation() >> quick_test.py
)

REM Kullanım bilgileri
echo %GREEN%
echo =================================================
echo    🎉 KURULUM TAMAMLANDI! (ensurepip Fix)
echo =================================================
echo %NC%

echo %BLUE%📖 Kullanım:%NC%
echo 1. Sanal ortamı aktive edin:
echo    %YELLOW%!VENV_NAME!\Scripts\activate.bat%NC%
echo.
echo 2. Hızlı test çalıştırın:
echo    %YELLOW%python quick_test.py%NC%
echo.
echo 3. Hafta 3 modüllerini çalıştırın:
echo    %YELLOW%python 01_autotokenizer_automodel.py%NC%
echo    %YELLOW%python 02_gpt_bert_t5_comparison.py%NC%
echo    %YELLOW%python 03_cpu_gpu_optimization.py%NC%
echo    %YELLOW%python 04_performance_measurement.py%NC%
echo.
echo 4. Jupyter notebook başlatın:
echo    %YELLOW%jupyter notebook%NC%
echo.

echo %GREEN%🔧 Sistem Bilgileri:%NC%
echo Python: !PYTHON_VERSION!
echo GPU Desteği: !GPU_TYPE!
echo Sanal Ortam: !VENV_NAME!
echo pip Yükleme: Manuel (ensurepip bypass)
echo.

echo %YELLOW%💡 İpuçları:%NC%
echo • Sanal ortamdan çıkmak için: %YELLOW%deactivate%NC%
echo • GPU memory temizlemek için: %YELLOW%python -c "import torch; torch.cuda.empty_cache()"%NC%
echo • Paket güncellemek için: %YELLOW%pip install --upgrade ^<paket_adı^>%NC%
echo.

echo %GREEN%🚀 İyi çalışmalar!%NC%
echo.
echo %BLUE%Sanal ortam aktif kalıyor... Bu pencereyi kapatmayın.%NC%

REM Sanal ortamı aktif bırak
cmd /k