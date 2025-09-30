"""
Pipeline ile GPU/CPU Performansını Ölçme ve Kıyaslama

Bu modül pipeline'lar kullanarak GPU ve CPU performansını ölçme,
karşılaştırma ve analiz etme yöntemlerini gösterir.
"""

import time
import torch
import psutil
import json
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Any

class PerformanceMeter:
    """Performans ölçümleri için yardımcı sınıf"""
    
    def __init__(self):
        self.results = []
        self.current_measurement = {}
    
    def start_measurement(self, name: str, device: str, model: str):
        """Ölçüm başlat"""
        self.current_measurement = {
            'name': name,
            'device': device,
            'model': model,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
    
    def end_measurement(self):
        """Ölçümü sonlandır"""
        if not self.current_measurement:
            return
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        result = {
            **self.current_measurement,
            'end_time': end_time,
            'duration': end_time - self.current_measurement['start_time'],
            'end_memory': end_memory,
            'memory_diff': self._calculate_memory_diff(
                self.current_measurement['start_memory'], 
                end_memory
            )
        }
        
        self.results.append(result)
        self.current_measurement = {}
        return result
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Bellek kullanımını al"""
        memory_info = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1e9
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
        
        return memory_info
    
    def _calculate_memory_diff(self, start: Dict, end: Dict) -> Dict[str, float]:
        """Bellek farkını hesapla"""
        diff = {}
        for key in start.keys():
            if key in end:
                diff[f"{key}_diff"] = end[key] - start[key]
        return diff
    
    def get_summary(self) -> Dict[str, Any]:
        """Özet istatistikleri al"""
        if not self.results:
            return {}
        
        summary = {
            'total_measurements': len(self.results),
            'devices': list(set(r['device'] for r in self.results)),
            'models': list(set(r['model'] for r in self.results)),
            'avg_duration_by_device': {},
            'avg_duration_by_model': {}
        }
        
        # Device bazında ortalama süre
        for device in summary['devices']:
            device_results = [r for r in self.results if r['device'] == device]
            avg_duration = np.mean([r['duration'] for r in device_results])
            summary['avg_duration_by_device'][device] = avg_duration
        
        # Model bazında ortalama süre
        for model in summary['models']:
            model_results = [r for r in self.results if r['model'] == model]
            avg_duration = np.mean([r['duration'] for r in model_results])
            summary['avg_duration_by_model'][model] = avg_duration
        
        return summary

def get_available_devices() -> List[str]:
    """Mevcut cihazları listele"""
    devices = ["cpu"]
    
    if torch.cuda.is_available():
        devices.append("cuda")
    
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    return devices

def test_sentiment_analysis_performance():
    """Sentiment analysis performansını test et"""
    print("=== Sentiment Analysis Performans Testi ===")
    
    performance_meter = PerformanceMeter()
    devices = get_available_devices()
    
    # Test verileri
    test_texts = [
        "I love this product!",
        "This is terrible.",
        "Not bad, could be better.",
        "Absolutely amazing experience!",
        "I'm not sure about this.",
        "Pretty good overall.",
        "Disappointing quality.",
        "Exceeded my expectations!",
        "Average at best.",
        "Outstanding performance!"
    ]
    
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    for device in devices:
        print(f"\nDevice: {device.upper()}")
        
        try:
            # Pipeline oluştur
            if device == "cpu":
                classifier = pipeline("sentiment-analysis", model=model_name)
            else:
                classifier = pipeline("sentiment-analysis", model=model_name, device=device)
            
            # Warmup
            classifier(test_texts[0])
            
            # Ölçüm başlat
            performance_meter.start_measurement(
                name="sentiment_analysis",
                device=device,
                model=model_name
            )
            
            # Test çalıştır
            results = classifier(test_texts)
            
            # Ölçüm sonlandır
            measurement = performance_meter.end_measurement()
            
            print(f"  Süre: {measurement['duration']:.4f} saniye")
            print(f"  Ortalama per text: {measurement['duration'] / len(test_texts):.4f} saniye")
            print(f"  CPU kullanımı: {measurement['end_memory']['cpu_percent']:.1f}%")
            print(f"  RAM kullanımı: {measurement['end_memory']['ram_used_gb']:.2f} GB")
            
            if device in ["cuda", "mps"] and "gpu_allocated_gb" in measurement['end_memory']:
                print(f"  GPU bellek: {measurement['end_memory']['gpu_allocated_gb']:.3f} GB")
        
        except Exception as e:
            print(f"  Hata: {str(e)}")
    
    return performance_meter

def test_text_generation_performance():
    """Text generation performansını test et"""
    print("\n=== Text Generation Performans Testi ===")
    
    performance_meter = PerformanceMeter()
    devices = get_available_devices()
    
    prompts = [
        "The future of technology is",
        "Machine learning will help us",
        "The most important skill for developers",
        "Artificial intelligence can improve",
        "The best way to learn programming"
    ]
    
    model_name = "gpt2"
    
    for device in devices:
        print(f"\nDevice: {device.upper()}")
        
        try:
            # Pipeline oluştur
            if device == "cpu":
                generator = pipeline("text-generation", model=model_name)
            else:
                generator = pipeline("text-generation", model=model_name, device=device)
            
            # Warmup
            generator(prompts[0], max_length=20, num_return_sequences=1, do_sample=False)
            
            # Ölçüm başlat
            performance_meter.start_measurement(
                name="text_generation",
                device=device,
                model=model_name
            )
            
            # Test çalıştır
            for prompt in prompts:
                result = generator(
                    prompt, 
                    max_length=50, 
                    num_return_sequences=1, 
                    do_sample=False,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
            
            # Ölçüm sonlandır
            measurement = performance_meter.end_measurement()
            
            print(f"  Süre: {measurement['duration']:.4f} saniye")
            print(f"  Ortalama per prompt: {measurement['duration'] / len(prompts):.4f} saniye")
            print(f"  CPU kullanımı: {measurement['end_memory']['cpu_percent']:.1f}%")
            print(f"  RAM kullanımı: {measurement['end_memory']['ram_used_gb']:.2f} GB")
            
        except Exception as e:
            print(f"  Hata: {str(e)}")
    
    return performance_meter

def test_question_answering_performance():
    """Question answering performansını test et"""
    print("\n=== Question Answering Performans Testi ===")
    
    performance_meter = PerformanceMeter()
    devices = get_available_devices()
    
    # Test verileri
    qa_pairs = [
        {
            "context": "Python is a programming language. It was created by Guido van Rossum.",
            "question": "Who created Python?"
        },
        {
            "context": "Machine learning is a subset of AI. It uses algorithms to learn from data.",
            "question": "What does machine learning use to learn?"
        },
        {
            "context": "The capital of France is Paris. It is famous for the Eiffel Tower.",
            "question": "What is the capital of France?"
        }
    ]
    
    model_name = "distilbert-base-cased-distilled-squad"
    
    for device in devices:
        print(f"\nDevice: {device.upper()}")
        
        try:
            # Pipeline oluştur
            if device == "cpu":
                qa_pipeline = pipeline("question-answering", model=model_name)
            else:
                qa_pipeline = pipeline("question-answering", model=model_name, device=device)
            
            # Warmup
            qa_pipeline(question=qa_pairs[0]["question"], context=qa_pairs[0]["context"])
            
            # Ölçüm başlat
            performance_meter.start_measurement(
                name="question_answering",
                device=device,
                model=model_name
            )
            
            # Test çalıştır
            for qa_pair in qa_pairs:
                result = qa_pipeline(
                    question=qa_pair["question"], 
                    context=qa_pair["context"]
                )
            
            # Ölçüm sonlandır
            measurement = performance_meter.end_measurement()
            
            print(f"  Süre: {measurement['duration']:.4f} saniye")
            print(f"  Ortalama per QA: {measurement['duration'] / len(qa_pairs):.4f} saniye")
            print(f"  CPU kullanımı: {measurement['end_memory']['cpu_percent']:.1f}%")
            
        except Exception as e:
            print(f"  Hata: {str(e)}")
    
    return performance_meter

def batch_size_performance_test():
    """Batch size performans testi"""
    print("\n=== Batch Size Performans Testi ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_sizes = [1, 4, 8, 16, 32]
    
    # Test verileri
    base_texts = [
        "This is a test sentence.",
        "Another example for testing.",
        "Performance analysis with batches.",
        "Batch processing optimization.",
        "Speed improvement measurement."
    ]
    
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    if device == "cpu":
        classifier = pipeline("sentiment-analysis", model=model_name)
    else:
        classifier = pipeline("sentiment-analysis", model=model_name, device=device)
    
    print(f"Device: {device.upper()}")
    print("Batch Size | Duration | Throughput (texts/sec)")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        # Veri hazırla
        texts = base_texts * (batch_size // len(base_texts) + 1)
        texts = texts[:batch_size]
        
        try:
            # Warmup
            classifier(texts[0])
            
            # Test
            start_time = time.time()
            results = classifier(texts)
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = len(texts) / duration
            
            print(f"{batch_size:10d} | {duration:8.4f} | {throughput:16.2f}")
            
        except Exception as e:
            print(f"{batch_size:10d} | Error: {str(e)[:20]}")
    
    print()

def model_comparison_benchmark():
    """Model karşılaştırma benchmark'ı"""
    print("\n=== Model Karşılaştırma Benchmark ===")
    
    models_to_test = [
        ("DistilBERT", "distilbert-base-uncased-finetuned-sst-2-english"),
        ("BERT-base", "nlptown/bert-base-multilingual-uncased-sentiment"),
        ("RoBERTa", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_text = "This is an amazing product with excellent quality!"
    
    print(f"Device: {device.upper()}")
    print("Model        | Load Time | Inference Time | Memory Usage")
    print("-" * 60)
    
    for model_name, model_path in models_to_test:
        try:
            # Model yükleme süresi
            start_time = time.time()
            if device == "cpu":
                classifier = pipeline("sentiment-analysis", model=model_path)
            else:
                classifier = pipeline("sentiment-analysis", model=model_path, device=device)
            load_time = time.time() - start_time
            
            # Inference süresi
            start_time = time.time()
            for _ in range(10):  # 10 tekrar
                result = classifier(test_text)
            inference_time = (time.time() - start_time) / 10
            
            # Bellek kullanımı
            memory_usage = psutil.virtual_memory().used / 1e9
            
            print(f"{model_name:12s} | {load_time:9.3f} | {inference_time:14.4f} | {memory_usage:11.2f} GB")
            
        except Exception as e:
            print(f"{model_name:12s} | Error: {str(e)[:40]}")
    
    print()

def create_performance_report(performance_meters: List[PerformanceMeter]):
    """Performans raporu oluştur"""
    print("\n=== Performans Raporu ===")
    
    all_results = []
    for meter in performance_meters:
        all_results.extend(meter.results)
    
    if not all_results:
        print("Henüz ölçüm verisi yok.")
        return
    
    # Device bazında özetleme
    devices = list(set(r['device'] for r in all_results))
    tasks = list(set(r['name'] for r in all_results))
    
    print("\nDevice Performans Özeti:")
    print("Device | Avg Duration | Task Count")
    print("-" * 35)
    
    for device in devices:
        device_results = [r for r in all_results if r['device'] == device]
        avg_duration = np.mean([r['duration'] for r in device_results])
        task_count = len(device_results)
        
        print(f"{device:6s} | {avg_duration:12.4f} | {task_count:10d}")
    
    # Task bazında özetleme
    print("\nTask Performans Özeti:")
    print("Task               | Avg Duration | Device Count")
    print("-" * 50)
    
    for task in tasks:
        task_results = [r for r in all_results if r['name'] == task]
        avg_duration = np.mean([r['duration'] for r in task_results])
        device_count = len(set(r['device'] for r in task_results))
        
        print(f"{task:18s} | {avg_duration:12.4f} | {device_count:12d}")
    
    # En hızlı kombinasyonlar
    print("\nEn Hızlı Kombinasyonlar:")
    sorted_results = sorted(all_results, key=lambda x: x['duration'])
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {result['name']} on {result['device']}: {result['duration']:.4f}s")
    
    print()

def save_benchmark_results(performance_meters: List[PerformanceMeter], filename: str = "benchmark_results.json"):
    """Benchmark sonuçlarını kaydet"""
    all_results = []
    for meter in performance_meters:
        all_results.extend(meter.results)
    
    # JSON serializable format'a çevir
    serializable_results = []
    for result in all_results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                serializable_result[key] = value
            else:
                serializable_result[key] = str(value)
        serializable_results.append(serializable_result)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Benchmark sonuçları {filename} dosyasına kaydedildi.")

def performance_tips():
    """Performans ipuçları"""
    print("\n=== Performans Optimizasyon İpuçları ===")
    
    tips = [
        "🚀 GPU Optimizasyonu:",
        "   • Uygun batch size kullanın (GPU memory'ye göre)",
        "   • Model ve data'yı aynı device'da tutun",
        "   • torch.cuda.empty_cache() ile memory temizleyin",
        "",
        "💾 Bellek Optimizasyonu:",
        "   • torch.no_grad() kullanın inference için",
        "   • Gereksiz gradient calculation'ı kapatın",
        "   • Model quantization (8-bit, 16-bit) uygulayın",
        "",
        "⚡ Hız Optimizasyonu:",
        "   • Pipeline'ları device ile optimize edin",
        "   • Warmup yapın ilk inference'dan önce",
        "   • Batch processing kullanın",
        "",
        "📊 Ölçüm İpuçları:",
        "   • Warmup yaparak JIT compilation etkisini azaltın",
        "   • Birden fazla çalıştırma yapın ve ortalama alın",
        "   • Memory ve CPU kullanımını da ölçün",
        "",
        "🔧 Model Seçimi:",
        "   • Küçük modeller (DistilBERT) hızlı inference",
        "   • Büyük modeller (BERT-large) daha iyi accuracy",
        "   • Task'a uygun model seçin"
    ]
    
    for tip in tips:
        print(tip)
    
    print()

if __name__ == "__main__":
    print("Pipeline ile GPU/CPU Performansını Ölçme ve Kıyaslama\n")
    
    # Performans testleri
    performance_meters = []
    
    # 1. Sentiment Analysis
    meter1 = test_sentiment_analysis_performance()
    performance_meters.append(meter1)
    
    # 2. Text Generation
    meter2 = test_text_generation_performance()
    performance_meters.append(meter2)
    
    # 3. Question Answering
    meter3 = test_question_answering_performance()
    performance_meters.append(meter3)
    
    # 4. Batch Size Testi
    batch_size_performance_test()
    
    # 5. Model Karşılaştırması
    model_comparison_benchmark()
    
    # 6. Performans Raporu
    create_performance_report(performance_meters)
    
    # 7. Sonuçları kaydet
    save_benchmark_results(performance_meters)
    
    # 8. İpuçları
    performance_tips()
    
    print("Performans ölçümleri tamamlandı!")