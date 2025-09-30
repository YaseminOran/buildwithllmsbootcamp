#!/bin/bash

# Build with LLMs Bootcamp - Kurulum ve Başlatma Scripti
# Bu script sanal ortam oluşturur ve tüm bağımlılıkları yükler

set -e  # Herhangi bir hata durumunda scripti durdur

# Renkli output için
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logo ve başlık
echo -e "${BLUE}"
echo "================================================="
echo "   🚀 BUILD WITH LLMS BOOTCAMP KURULUM (MAC/LINUX)"
echo "================================================="
echo -e "${NC}"

# Platform kontrolü
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo -e "${YELLOW}⚠️  Windows tespit edildi!${NC}"
    echo -e "${BLUE}Lütfen start.bat dosyasını kullanın:${NC}"
    echo -e "  ${YELLOW}start.bat${NC}"
    echo
    exit 0
fi

# Gereksinimler kontrolü
echo -e "${YELLOW}📋 Sistem gereksinimleri kontrol ediliyor...${NC}"

# Python versiyonu kontrolü
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 bulunamadı. Lütfen Python 3.8+ yükleyin.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python $PYTHON_VERSION bulundu${NC}"

# pip kontrolü
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 bulunamadı. Lütfen pip yükleyin.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ pip3 bulundu${NC}"

# Git kontrolü
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}⚠️  Git bulunamadı. Git yüklemeniz önerilir.${NC}"
else
    echo -e "${GREEN}✅ Git bulundu${NC}"
fi

# GPU desteği kontrolü
echo -e "${YELLOW}🔍 GPU desteği kontrol ediliyor...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU bulundu${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    GPU_TYPE="cuda"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - Apple Silicon kontrolü
    if [[ $(uname -m) == "arm64" ]]; then
        echo -e "${GREEN}✅ Apple Silicon (M1/M2) bulundu - MPS desteği mevcut${NC}"
        GPU_TYPE="mps"
    else
        echo -e "${YELLOW}⚠️  Intel Mac - Sadece CPU desteği${NC}"
        GPU_TYPE="cpu"
    fi
else
    echo -e "${YELLOW}⚠️  GPU bulunamadı - CPU modunda çalışacak${NC}"
    GPU_TYPE="cpu"
fi

# Sanal ortam oluşturma
VENV_NAME="llm_bootcamp_env"
echo -e "${YELLOW}🔧 Sanal ortam oluşturuluyor: $VENV_NAME${NC}"

if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}⚠️  Sanal ortam zaten mevcut. Yeniden oluşturuluyor...${NC}"
    rm -rf "$VENV_NAME"
fi

python3 -m venv "$VENV_NAME"
echo -e "${GREEN}✅ Sanal ortam oluşturuldu${NC}"

# Sanal ortamı aktive et
echo -e "${YELLOW}🔄 Sanal ortam aktive ediliyor...${NC}"
source "$VENV_NAME/bin/activate"

# pip'i güncelle
echo -e "${YELLOW}⬆️  pip güncelleniyor...${NC}"
pip install --upgrade pip setuptools wheel

# PyTorch yükleme - GPU tipine göre
echo -e "${YELLOW}🔥 PyTorch yükleniyor...${NC}"

if [ "$GPU_TYPE" == "cuda" ]; then
    echo -e "${BLUE}CUDA desteği ile PyTorch yükleniyor...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$GPU_TYPE" == "mps" ]; then
    echo -e "${BLUE}Apple Silicon için PyTorch yükleniyor...${NC}"
    pip install torch torchvision torchaudio
else
    echo -e "${BLUE}CPU versiyonu PyTorch yükleniyor...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✅ PyTorch yüklendi${NC}"

# requirements.txt'den diğer paketleri yükle
echo -e "${YELLOW}📦 Diğer bağımlılıklar yükleniyor...${NC}"

# GPU tipine göre requirements dosyasını güncelle
if [ "$GPU_TYPE" == "cpu" ]; then
    # CPU için bazı GPU-specific paketleri çıkar
    grep -v "bitsandbytes\|GPUtil" requirements.txt > requirements_cpu.txt
    pip install -r requirements_cpu.txt
    rm requirements_cpu.txt
else
    pip install -r requirements.txt
fi

echo -e "${GREEN}✅ Tüm bağımlılıklar yüklendi${NC}"

# Spacy modeli yükleme
echo -e "${YELLOW}🔤 Spacy dil modeli yükleniyor...${NC}"
python -m spacy download en_core_web_sm

# NLTK verileri yükleme
echo -e "${YELLOW}📚 NLTK verileri yükleniyor...${NC}"
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print('NLTK verileri yüklendi')
except:
    print('NLTK verilerini yüklerken hata oluştu')
"

# Kurulum testi
echo -e "${YELLOW}🧪 Kurulum test ediliyor...${NC}"

python3 -c "
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ NumPy: {np.__version__}')

# GPU testi
if torch.cuda.is_available():
    print(f'✅ CUDA: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('✅ Apple MPS destekleniyor')
else:
    print('✅ CPU modunda çalışıyor')
"

# Hızlı test scripti oluştur
echo -e "${YELLOW}📝 Test scripti oluşturuluyor...${NC}"

cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
Hızlı kurulum testi
"""

import torch
from transformers import pipeline
import time

def test_installation():
    print("🧪 Kurulum testi başlatılıyor...")
    
    # Device kontrolü
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple MPS destekleniyor")
    else:
        device = "cpu"
        print("✅ CPU modunda çalışıyor")
    
    # Basit sentiment analysis testi
    print("\n🔍 Sentiment analysis testi...")
    classifier = pipeline("sentiment-analysis")
    
    start_time = time.time()
    result = classifier("This is a great bootcamp!")
    end_time = time.time()
    
    print(f"Sonuç: {result[0]['label']} (güven: {result[0]['score']:.4f})")
    print(f"Süre: {end_time - start_time:.4f} saniye")
    
    print("\n🎉 Test başarılı! Sistem kullanıma hazır.")

if __name__ == "__main__":
    test_installation()
EOF

chmod +x quick_test.py

# Kullanım bilgileri
echo -e "${GREEN}"
echo "================================================="
echo "   🎉 KURULUM TAMAMLANDI!"
echo "================================================="
echo -e "${NC}"

echo -e "${BLUE}📖 Kullanım:${NC}"
echo -e "1. Sanal ortamı aktive edin:"
echo -e "   ${YELLOW}source $VENV_NAME/bin/activate${NC}"
echo -e ""
echo -e "2. Hızlı test çalıştırın:"
echo -e "   ${YELLOW}python quick_test.py${NC}"
echo -e ""
echo -e "3. Hafta 3 modüllerini çalıştırın:"
echo -e "   ${YELLOW}python 01_autotokenizer_automodel.py${NC}"
echo -e "   ${YELLOW}python 02_gpt_bert_t5_comparison.py${NC}"
echo -e "   ${YELLOW}python 03_cpu_gpu_optimization.py${NC}"
echo -e "   ${YELLOW}python 04_performance_measurement.py${NC}"
echo -e ""
echo -e "4. Jupyter notebook başlatın:"
echo -e "   ${YELLOW}jupyter notebook${NC}"
echo -e ""

echo -e "${GREEN}🔧 Sistem Bilgileri:${NC}"
echo -e "Python: $PYTHON_VERSION"
echo -e "GPU Desteği: $GPU_TYPE"
echo -e "Sanal Ortam: $VENV_NAME"
echo -e ""

echo -e "${YELLOW}💡 İpuçları:${NC}"
echo -e "• Sanal ortamdan çıkmak için: ${YELLOW}deactivate${NC}"
echo -e "• GPU memory temizlemek için: ${YELLOW}python -c 'import torch; torch.cuda.empty_cache()'${NC}"
echo -e "• Paket güncellemek için: ${YELLOW}pip install --upgrade <paket_adı>${NC}"
echo -e ""

echo -e "${GREEN}🚀 İyi çalışmalar!${NC}"

# Sanal ortamı aktive bırak
echo -e "${BLUE}Sanal ortam aktif kalıyor...${NC}"