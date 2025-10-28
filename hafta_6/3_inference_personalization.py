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
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, 
                     deterministic=False):
        """
        Text generation with improved tokenization and parameters
        """
        # Proper tokenization with attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,  # Only new tokens, not total
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if deterministic:
            # Deterministic generation for factual/technical content
            generation_kwargs.update({
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            })
        else:
            # Creative generation
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": 50,
            })
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generated_text.strip()
    
    def classify_text(self, text, label_names=None):
        """
        Text classification with proper output formatting
        """
        # Proper tokenization
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probabilities = predictions.cpu().numpy()[0]
            predicted_class = probabilities.argmax()
        
        # Format output with labels
        result = {
            "predicted_class": int(predicted_class),
            "probabilities": probabilities.tolist(),
            "confidence": float(probabilities[predicted_class])
        }
        
        if label_names:
            result["predicted_label"] = label_names[predicted_class]
            result["all_predictions"] = {
                label_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        
        return result

def load_lora_model(base_model_path, lora_adapter_path, merge_adapters=False):
    """
    LoRA adapter ile model yükleme (geliştirilmiş)
    """
    print("LoRA modeli yükleniyor...")
    
    # Base model with device mapping
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",  # Automatic device placement
        torch_dtype="auto"  # Automatic dtype selection
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA adapter yükleme
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()  # Set to evaluation mode
    
    # Optional: merge adapters for faster inference
    if merge_adapters:
        print("LoRA adapter'ları base model ile birleştiriliyor...")
        model = model.merge_and_unload()
    
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

def load_quantized_model(model_path, quantization="8bit"):
    """
    Quantized model yükleme örneği
    """
    print(f"Model {quantization} quantization ile yükleniyor...")
    
    if quantization == "8bit":
        # 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
    elif quantization == "4bit":
        # 4-bit quantization (requires bitsandbytes)
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        # Standard loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

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
# Hugging Face Pipeline kullanımı (iyileştirilmiş)
from transformers import pipeline, AutoTokenizer

# Tokenizer yükle ve pad token ayarla
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Text generation pipeline
generator = pipeline(
    "text-generation",
    model="./fine_tuned_model",
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype="auto"
)

# Kullanım - deterministik mod
result_factual = generator(
    "Yapay zeka nedir?",
    max_new_tokens=50,
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id
)

# Kullanım - yaratıcı mod  
result_creative = generator(
    "Bir zamanlar...",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
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
    
    # Geliştirilmiş örnekler
    print("\n=== Pratik Kullanım Örnekleri ===")
    
    # Deterministik vs Yaratıcı örneği
    example_usage = '''
# Deterministik üretim örneği (teknik/faktüel içerik için)
factual_result = inference_engine.generate_text(
    "Python'da liste nedir?",
    max_new_tokens=50,
    deterministic=True  # Tutarlı, tekrarlanabilir sonuçlar
)

# Yaratıcı üretim örneği (hikaye/blog için)  
creative_result = inference_engine.generate_text(
    "Bir zamanlar uzak bir galakside...",
    max_new_tokens=100,
    temperature=0.8,
    deterministic=False  # Yaratıcı, çeşitli sonuçlar
)

# Sınıflandırma örneği (etiket adları ile)
classification_result = inference_engine.classify_text(
    "Bu film gerçekten harikaydi!",
    label_names=["Negatif", "Pozitif"]
)
print(f"Tahmin: {classification_result['predicted_label']}")
print(f"Güven: {classification_result['confidence']:.2f}")
'''
    print(example_usage)
    
    # Gerçek model inference örneği (uncomment to run)
    # model_path = "./fine_tuned_model"  # Trained model path
    # inference_engine = PersonalizedInference(model_path)
    # result = inference_engine.generate_text("Merhaba, nasılsın?", deterministic=False)
    # print(f"Generated: {result}")