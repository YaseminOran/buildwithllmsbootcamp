"""
CPU/GPU Performans Yönetimi ve Model Optimizasyonu

Bu modül CPU ve GPU performansını yönetme, model optimizasyonu teknikleri
ve farklı cihazlar arasında performans karşılaştırması yapma yöntemlerini gösterir.
"""

import torch
torch.backends.quantized.engine = "qnnpack"
import time
import psutil
import gc
from transformers import (
    AutoTokenizer, AutoModel, pipeline,
    BitsAndBytesConfig, AutoModelForSequenceClassification
)
import numpy as np

def check_device_availability():
    """Mevcut cihazları kontrol et"""
    print("=== Cihaz Durumu ===")
    print(f"PyTorch versiyonu: {torch.__version__}")
    print(f"CUDA mevcut: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA cihaz sayısı: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Bellek: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print(f"MPS (Apple Silicon) mevcut: {torch.backends.mps.is_available()}")
    print(f"CPU çekirdek sayısı: {psutil.cpu_count()}")
    print(f"Sistem RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print()

def get_optimal_device():
    """En uygun cihazı seç"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def memory_monitoring():
    """Bellek kullanımını izle"""
    def get_memory_usage():
        """Mevcut bellek kullanımını al"""
        memory_info = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available": psutil.virtual_memory().available / 1e9
        }
        
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
            memory_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        
        return memory_info
    
    return get_memory_usage

def cpu_optimization_example():
    """CPU optimizasyonu örnekleri"""
    print("=== CPU Optimizasyonu ===")
    
    # CPU thread sayısını ayarla
    original_threads = torch.get_num_threads()
    print(f"Varsayılan thread sayısı: {original_threads}")
    
    # Optimal thread sayısını hesapla
    cpu_count = psutil.cpu_count(logical=False)  # Fiziksel çekirdek sayısı
    optimal_threads = min(cpu_count, 4)  # Maksimum 4 thread
    
    torch.set_num_threads(optimal_threads)
    print(f"Optimized thread sayısı: {optimal_threads}")
    
    # Test modeli
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # CPU'da çalıştır
    device = torch.device("cpu")
    model = model.to(device)
    
    text = "This is a test sentence for CPU optimization."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Performans testi
    memory_monitor = memory_monitoring()
    
    start_memory = memory_monitor()
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(10):
            outputs = model(**inputs)
    
    end_time = time.time()
    end_memory = memory_monitor()
    
    print(f"CPU inference süresi (10 iterasyon): {end_time - start_time:.4f} saniye")
    print(f"Ortalama inference süresi: {(end_time - start_time) / 10:.4f} saniye")
    print(f"CPU kullanımı: {end_memory['cpu_percent']:.1f}%")
    print(f"RAM kullanımı: {end_memory['ram_percent']:.1f}%")
    
    # Thread sayısını geri al
    torch.set_num_threads(original_threads)
    print()

def gpu_optimization_example():
    """GPU optimizasyonu örnekleri"""
    print("=== GPU Optimizasyonu ===")
    
    device = get_optimal_device()
    print(f"Kullanılan cihaz: {device}")
    
    if device.type == "cpu":
        print("GPU mevcut değil, CPU kullanılıyor.")
        return
    
    # Model ve tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # GPU'ya taşı
    model = model.to(device)
    
    text = "This is a test sentence for GPU optimization."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # GPU memory temizle
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU bellek kullanımı (başlangıç): {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    memory_monitor = memory_monitoring()
    start_memory = memory_monitor()
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(100):  # GPU için daha fazla iterasyon
            outputs = model(**inputs)
    
    end_time = time.time()
    end_memory = memory_monitor()
    
    print(f"GPU inference süresi (100 iterasyon): {end_time - start_time:.4f} saniye")
    print(f"Ortalama inference süresi: {(end_time - start_time) / 100:.4f} saniye")
    
    if device.type == "cuda":
        print(f"GPU bellek kullanımı (son): {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"GPU bellek rezervi: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    
    print()

def model_quantization_example():
    """Model quantization örneği"""
    print("=== Model Quantization ===")
    
    model_name = "distilbert-base-uncased"
    
    # Normal model
    print("1. Normal Model (FP32):")
    model_fp32 = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Model boyutunu hesapla
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1e6  # MB
    
    fp32_size = get_model_size(model_fp32)
    print(f"FP32 model boyutu: {fp32_size:.2f} MB")
    
    # Quantized model (8-bit)
    try:
        print("\n2. Quantized Model (8-bit):")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        model_8bit = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        print("8-bit quantization başarılı!")
        print("Model boyutu yaklaşık %75 azaldı")
        
    except Exception as e:
        print(f"8-bit quantization desteklenmiyor: {str(e)}")
        print("BitsAndBytesConfig kurulumu gerekli olabilir")
    
    # Dynamic quantization
    print("\n3. Dynamic Quantization:")
    model_dynamic = torch.quantization.quantize_dynamic(
        model_fp32, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    dynamic_size = get_model_size(model_dynamic)
    print(f"Dynamic quantized model boyutu: {dynamic_size:.2f} MB")
    print(f"Boyut azalması: {((fp32_size - dynamic_size) / fp32_size * 100):.1f}%")
    
    # Performans karşılaştırması
    text = "Test sentence for quantization performance."
    inputs = tokenizer(text, return_tensors="pt")
    
    models_to_test = [
        ("FP32", model_fp32),
        ("Dynamic Quantized", model_dynamic)
    ]
    
    print("\nPerformans karşılaştırması:")
    for name, model in models_to_test:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(50):
                outputs = model(**inputs)
        end_time = time.time()
        
        print(f"{name}: {(end_time - start_time):.4f} saniye")
    
    print()

def batch_processing_optimization():
    """Batch processing optimizasyonu"""
    print("=== Batch Processing Optimizasyonu ===")
    
    device = get_optimal_device()
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Test verileri
    texts = [
        "This is the first test sentence.",
        "Here is another example text.",
        "Batch processing improves efficiency.",
        "Parallel processing saves time.",
        "Optimization is crucial for performance."
    ] * 4  # 20 text toplam
    
    # Tek tek işleme
    print("1. Tek tek işleme:")
    start_time = time.time()
    
    individual_results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            individual_results.append(outputs.last_hidden_state)
    
    individual_time = time.time() - start_time
    
    # Batch işleme
    print("2. Batch işleme:")
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        start_time = time.time()
        batch_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", 
                             padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_results.append(outputs.last_hidden_state)
        
        batch_time = time.time() - start_time
        speedup = individual_time / batch_time
        
        print(f"  Batch size {batch_size:2d}: {batch_time:.4f}s (speedup: {speedup:.2f}x)")
    
    print(f"Tek tek işleme: {individual_time:.4f}s")
    print()

def memory_efficient_inference():
    """Bellek verimli inference"""
    print("=== Bellek Verimli Inference ===")
    
    device = get_optimal_device()
    model_name = "distilbert-base-uncased"
    
    # Gradient calculation'ı kapat
    print("1. Gradient Calculation Kapalı:")
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "Memory efficient inference example."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    memory_monitor = memory_monitoring()
    
    # With gradients
    start_memory = memory_monitor()
    start_time = time.time()
    
    for _ in range(10):
        outputs = model(**inputs)  # Gradients enabled
    
    with_grad_time = time.time() - start_time
    with_grad_memory = memory_monitor()
    
    # Without gradients
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(**inputs)
    
    no_grad_time = time.time() - start_time
    no_grad_memory = memory_monitor()
    
    print(f"Gradient ile: {with_grad_time:.4f}s")
    print(f"Gradient olmadan: {no_grad_time:.4f}s")
    print(f"Hız artışı: {with_grad_time / no_grad_time:.2f}x")
    
    # Memory cleanup
    print("\n2. Memory Cleanup:")
    if device.type == "cuda":
        print(f"GPU bellek (başlangıç): {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        
        # Manual cleanup
        del model, outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"GPU bellek (temizlik sonrası): {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    print()

def pipeline_device_optimization():
    """Pipeline device optimizasyonu"""
    print("=== Pipeline Device Optimizasyonu ===")
    
    device = get_optimal_device()
    
    # Pipeline with device specification
    if device.type != "cpu":
        print(f"GPU/MPS ile pipeline kullanımı ({device}):")
        
        # Sentiment analysis pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
        
        texts = [
            "This is amazing!",
            "I don't like this.",
            "Pretty good overall.",
            "Absolutely terrible experience.",
            "Could be better."
        ]
        
        start_time = time.time()
        results = classifier(texts)
        gpu_time = time.time() - start_time
        
        print(f"GPU/MPS inference süresi: {gpu_time:.4f}s")
        
        # CPU comparison
        classifier_cpu = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device="cpu"
        )
        
        start_time = time.time()
        results_cpu = classifier_cpu(texts)
        cpu_time = time.time() - start_time
        
        print(f"CPU inference süresi: {cpu_time:.4f}s")
        print(f"Hız artışı: {cpu_time / gpu_time:.2f}x")
        
    else:
        print("Sadece CPU mevcut, pipeline CPU'da çalışıyor.")
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        texts = ["This is a test sentence."]
        results = classifier(texts)
        print("CPU pipeline çalışması tamamlandı.")
    
    print()

def optimization_best_practices():
    """Optimizasyon en iyi uygulamaları"""
    print("=== Optimizasyon En İyi Uygulamaları ===")
    
    best_practices = [
        "✅ torch.no_grad() kullanın inference için",
        "✅ Uygun batch size seçin (GPU memory'ye göre)",
        "✅ Model quantization uygulayın (8-bit, dynamic)",
        "✅ Gereksiz gradient calculation'ı kapatın",
        "✅ Memory cleanup yapın (del, gc.collect(), torch.cuda.empty_cache())",
        "✅ Optimal device seçin (CUDA > MPS > CPU)",
        "✅ CPU thread sayısını ayarlayın",
        "✅ Padding'i minimize edin",
        "✅ Model caching kullanın",
        "✅ Pipeline'ları device ile optimize edin"
    ]
    
    for practice in best_practices:
        print(f"  {practice}")
    
    print("\n⚠️  Dikkat edilmesi gerekenler:")
    warnings = [
        "❌ Çok büyük batch size'lar memory overflow'a sebep olabilir",
        "❌ Quantization accuracy kaybına sebep olabilir",
        "❌ CPU'da çok fazla thread performansı düşürebilir",
        "❌ Memory leak'lere dikkat edin",
        "❌ Device transfer'ları minimize edin"
    ]
    
    for warning in warnings:
        print(f"  {warning}")
    
    print()

if __name__ == "__main__":
    print("CPU/GPU Performans Yönetimi ve Model Optimizasyonu\n")
    
    check_device_availability()
    cpu_optimization_example()
    gpu_optimization_example()
    model_quantization_example()
    batch_processing_optimization()
    memory_efficient_inference()
    pipeline_device_optimization()
    optimization_best_practices()
    
    print("Optimizasyon örnekleri tamamlandı!")