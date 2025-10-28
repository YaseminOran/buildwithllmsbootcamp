"""
Inference ve Kişiselleştirilmiş Model Kullanımı

Bu script, fine-tune edilmiş modelleri inference için kullanma
ve kişiselleştirilmiş çıktılar üretme konularını kapsar.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
    GenerationConfig
)
from peft import PeftModel
import time
import json

class PersonalizedInference:
    """
    Kişiselleştirilmiş inference için ana sınıf
    """
    
    def __init__(self, model_path, model_type="causal_lm"):
        self.model_path = model_path
        self.model_type = model_type
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Model ve tokenizer yükleme
        """
        print(f"Model yükleniyor: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.model_type == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        elif self.model_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # GPU varsa kullan
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model GPU'ya yüklendi")
        
        self.model.eval()
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """
        Text generation
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    
    def classify_text(self, text):
        """
        Text classification
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return predictions.cpu().numpy()[0]

def load_lora_model(base_model_path, lora_adapter_path):
    """
    LoRA adapter ile model yükleme
    """
    print("LoRA modeli yükleniyor...")
    
    # Base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # LoRA adapter yükleme
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    return model, tokenizer

def demonstrate_inference_optimization():
    """
    Inference optimizasyon tekniklerini gösterir
    """
    print("\n=== Inference Optimizasyon Teknikleri ===")
    
    techniques = {
        "Model Quantization": "INT8/FP16 precision kullanımı",
        "Batch Processing": "Birden fazla input'u aynı anda işleme",
        "Caching": "KV-cache ve attention cache kullanımı",
        "ONNX Export": "Model'i ONNX formatına çevirme",
        "TensorRT": "NVIDIA GPU'lar için optimizasyon",
        "Dynamic Batching": "Farklı uzunluktaki sequence'leri gruplandırma"
    }
    
    for technique, description in techniques.items():
        print(f"- {technique}: {description}")

def create_personalized_chatbot():
    """
    Kişiselleştirilmiş chatbot örneği
    """
    print("\n=== Kişiselleştirilmiş Chatbot ===")
    
    # Kullanıcı profili
    user_profile = {
        "name": "Ali",
        "interests": ["teknoloji", "yapay zeka", "python"],
        "style": "samimi ve yardımsever",
        "expertise_level": "intermediate"
    }
    
    def create_personalized_prompt(user_input, profile):
        """
        Kullanıcı profiline göre prompt oluşturma
        """
        prompt = f"""
Kullanıcı Profili:
- İsim: {profile['name']}
- İlgi Alanları: {', '.join(profile['interests'])}
- Konuşma Stili: {profile['style']}
- Seviye: {profile['expertise_level']}

{profile['name']}: {user_input}
Asistan:"""
        return prompt
    
    # Örnek conversation
    user_inputs = [
        "Merhaba! Python öğrenmek istiyorum.",
        "Makine öğrenmesi için hangi kütüphaneleri önerirsin?"
    ]
    
    for user_input in user_inputs:
        personalized_prompt = create_personalized_prompt(user_input, user_profile)
        print(f"\nPersonalized Prompt:\n{personalized_prompt}")
        print("-" * 50)

def demonstrate_generation_config():
    """
    Generation configuration seçeneklerini gösterir
    """
    print("\n=== Generation Configuration ===")
    
    configs = {
        "Creative Writing": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        },
        "Factual QA": {
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 10,
            "repetition_penalty": 1.0
        },
        "Code Generation": {
            "temperature": 0.2,
            "top_p": 0.6,
            "top_k": 20,
            "repetition_penalty": 1.05
        }
    }
    
    for use_case, config in configs.items():
        print(f"\n{use_case}:")
        for param, value in config.items():
            print(f"  {param}: {value}")

def benchmark_inference_speed():
    """
    Inference hızını ölçme
    """
    print("\n=== Inference Speed Benchmark ===")
    
    # Simulated timing
    scenarios = {
        "CPU - No Optimization": 1.5,
        "CPU - Quantized": 0.8,
        "GPU - FP32": 0.3,
        "GPU - FP16": 0.15,
        "GPU - INT8": 0.1
    }
    
    print("Scenario\t\t\tTime (seconds)")
    print("-" * 40)
    for scenario, time_taken in scenarios.items():
        print(f"{scenario:<25}\t{time_taken:.2f}s")

def create_inference_pipeline():
    """
    Inference pipeline oluşturma
    """
    print("\n=== Inference Pipeline Örneği ===")
    
    # Pipeline workflow
    steps = [
        "1. Input preprocessing",
        "2. Tokenization", 
        "3. Model inference",
        "4. Output postprocessing",
        "5. Response formatting"
    ]
    
    for step in steps:
        print(step)
    
    # Pipeline code örneği
    pipeline_example = '''
# Hugging Face Pipeline kullanımı
from transformers import pipeline

# Text generation pipeline
generator = pipeline(
    "text-generation",
    model="./fine_tuned_model",
    tokenizer="./fine_tuned_model",
    device=0 if torch.cuda.is_available() else -1
)

# Kullanım
result = generator(
    "Yapay zeka nedir?",
    max_length=100,
    temperature=0.7,
    pad_token_id=50256
)
'''
    print(f"\nPipeline Kod Örneği:\n{pipeline_example}")

def demonstrate_model_deployment():
    """
    Model deployment stratejilerini gösterir
    """
    print("\n=== Model Deployment Stratejileri ===")
    
    strategies = {
        "REST API": "Flask/FastAPI ile web servisi",
        "gRPC": "Yüksek performanslı RPC servisi",
        "Docker Container": "Containerized deployment",
        "Kubernetes": "Scalable cloud deployment",
        "Edge Deployment": "Mobile/IoT cihazlarda çalıştırma",
        "Serverless": "AWS Lambda, Google Cloud Functions"
    }
    
    for strategy, description in strategies.items():
        print(f"- {strategy}: {description}")
    
    # Deployment checklist
    print("\nDeployment Checklist:")
    checklist = [
        "✓ Model boyutu optimizasyonu",
        "✓ Inference hızı testi", 
        "✓ Memory kullanımı kontrolü",
        "✓ Error handling",
        "✓ Monitoring ve logging",
        "✓ Security considerations"
    ]
    
    for item in checklist:
        print(f"  {item}")

if __name__ == "__main__":
    print("Inference ve Kişiselleştirme Demonstrasyonu")
    print("=" * 50)
    
    # Tüm demonstrasyonları çalıştır
    demonstrate_inference_optimization()
    create_personalized_chatbot()
    demonstrate_generation_config()
    benchmark_inference_speed()
    create_inference_pipeline()
    demonstrate_model_deployment()
    
    print("\n" + "=" * 50)
    print("Demonstrasyon tamamlandı!")
    
    # Gerçek model inference örneği (uncomment to run)
    # model_path = "./fine_tuned_model"  # Trained model path
    # inference_engine = PersonalizedInference(model_path)
    # result = inference_engine.generate_text("Merhaba, nasılsın?")
    # print(f"Generated: {result}")