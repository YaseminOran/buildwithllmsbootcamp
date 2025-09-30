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
