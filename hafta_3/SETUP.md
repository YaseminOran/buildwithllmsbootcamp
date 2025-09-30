# 🚀 Hafta 3 - Kurulum Kılavuzu

Bu kılavuz, macOS, Linux ve Windows işletim sistemlerinde hafta 3 modüllerini çalıştırmak için gerekli kurulum adımlarını içerir.

## 🎯 Hızlı Başlangıç

### 🍎 macOS / 🐧 Linux

```bash
cd hafta_3
chmod +x start.sh
./start.sh
```

### 🪟 Windows

```cmd
cd hafta_3
start.bat
```

## 📋 Sistem Gereksinimleri

### Minimum Gereksinimler
- **Python**: 3.8 veya üzeri
- **RAM**: 8 GB (16 GB önerilir)
- **Disk Alanı**: 10 GB boş alan
- **İnternet**: Model indirme için

### GPU Desteği (Opsiyonel)
- **NVIDIA GPU**: CUDA 11.8+ ile uyumlu
- **Apple Silicon**: M1/M2 Mac'ler (MPS desteği)
- **GPU RAM**: Minimum 4 GB (8 GB+ önerilir)

## 🔧 Detaylı Kurulum

### 1. Python Kurulumu

#### macOS
```bash
# Homebrew ile
brew install python@3.11

# Veya python.org'dan indirin
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Windows
1. [python.org](https://python.org) adresinden Python indirin
2. Kurulum sırasında "Add Python to PATH" seçeneğini işaretleyin

### 2. Git Kurulumu (Opsiyonel)

#### macOS
```bash
brew install git
```

#### Linux
```bash
sudo apt install git
```

#### Windows
[git-scm.com](https://git-scm.com) adresinden Git indirin

### 3. GPU Sürücüleri (Opsiyonel)

#### NVIDIA GPU (Windows/Linux)
1. [NVIDIA Driver](https://www.nvidia.com/drivers/) indirin
2. [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-downloads) yükleyin

#### Apple Silicon (macOS)
Otomatik olarak desteklenir, ek kurulum gerektirmez.

## 🛠️ Manuel Kurulum

Otomatik script çalışmazsa manuel kurulum yapabilirsiniz:

### 1. Sanal Ortam Oluşturma

```bash
# macOS/Linux
python3 -m venv llm_bootcamp_env
source llm_bootcamp_env/bin/activate

# Windows CMD
python -m venv llm_bootcamp_env
llm_bootcamp_env\Scripts\activate.bat

# Windows PowerShell
python -m venv llm_bootcamp_env
llm_bootcamp_env\Scripts\Activate.ps1
```

### 2. pip Güncelleme

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. PyTorch Kurulumu

#### CUDA Destekli (NVIDIA GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CPU Versiyonu
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Apple Silicon (macOS)
```bash
pip install torch torchvision torchaudio
```

### 4. Diğer Bağımlılıklar

```bash
pip install -r requirements.txt
```

### 5. NLTK ve Spacy Verileri

```bash
# Spacy dil modeli
python -m spacy download en_core_web_sm

# NLTK verileri
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## 🧪 Kurulum Testi

### Hızlı Test
```bash
python quick_test.py
```

### Detaylı Test
```bash
python -c "
import torch
import transformers
import numpy as np

print(f'✅ Python: {__import__('sys').version}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ NumPy: {np.__version__}')

if torch.cuda.is_available():
    print(f'✅ CUDA: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('✅ Apple MPS destekleniyor')
else:
    print('✅ CPU modunda çalışıyor')
"
```

## 🔍 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### 1. "Python bulunamadı" Hatası
**Çözüm:**
- Python'un PATH'e eklendiğinden emin olun
- `python3` yerine `python` komutunu deneyin
- Python'u yeniden kurun

#### 2. "pip install" Hataları
**Çözüm:**
```bash
# pip'i güncelle
python -m pip install --upgrade pip

# Cache temizle
pip cache purge

# Proxy arkasındaysanız
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <paket>
```

#### 3. CUDA "Out of Memory" Hatası
**Çözüm:**
```python
import torch
torch.cuda.empty_cache()  # GPU memory temizle

# Batch size'ı azaltın
batch_size = 4  # 32 yerine
```

#### 4. Windows'ta "Execution Policy" Hatası
**Çözüm:**
```powershell
# PowerShell'i yönetici olarak açın
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 5. Apple Silicon Uyumluluk Sorunları
**Çözüm:**
```bash
# Rosetta ile çalıştırın
arch -x86_64 python -m pip install <paket>

# Veya native ARM64 versiyonunu kullanın
conda install <paket>
```

### Log Dosyalarını İnceleme

Kurulum sırasında hatalar oluşursa:

```bash
# Detaylı log ile kurulum
pip install -r requirements.txt --verbose > install.log 2>&1

# Log dosyasını inceleyin
cat install.log
```

## 📊 Performans Optimizasyonu

### RAM Optimizasyonu
```python
# Model cache boyutunu sınırla
import os
os.environ["TRANSFORMERS_CACHE"] = "./cache"
os.environ["HF_DATASETS_CACHE"] = "./cache"
```

### GPU Optimizasyonu
```python
# Mixed precision kullan
import torch
torch.backends.cudnn.benchmark = True
```

### CPU Optimizasyonu
```python
# Thread sayısını ayarla
import torch
torch.set_num_threads(4)
```

## 🌐 Ağ Gereksinimleri

### Model İndirme
İlk çalıştırmada modeller otomatik indirilir:
- **DistilBERT**: ~250 MB
- **BERT-base**: ~440 MB
- **GPT-2**: ~500 MB
- **T5-small**: ~240 MB

### Proxy Ayarları
Kurumsal ağ kullanıyorsanız:
```bash
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
pip install --proxy http://proxy:port <paket>
```

## 📞 Destek

### Hata Bildirimi
Sorun yaşıyorsanız aşağıdaki bilgileri toplayın:
1. İşletim sistemi ve versiyonu
2. Python versiyonu
3. Hata mesajının tam metni
4. `pip list` çıktısı

### Faydalı Komutlar
```bash
# Sistem bilgileri
python -m platform
python -c "import sys; print(sys.version)"

# Kurulu paketler
pip list

# GPU durumu
nvidia-smi  # NVIDIA için
```

## ✅ Başarılı Kurulum Kontrolü

Kurulum başarılıysa şunları görebilmelisiniz:
- ✅ Python 3.8+ yüklü
- ✅ Sanal ortam aktif
- ✅ PyTorch çalışıyor
- ✅ Transformers yüklü
- ✅ GPU destekleniyor (varsa)
- ✅ Test script başarıyla çalışıyor

---

**İyi çalışmalar! 🚀**