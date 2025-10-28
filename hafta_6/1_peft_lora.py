"""
PEFT (Parameter Efficient Fine-Tuning) ve LoRA Örneği

Bu script, LoRA kullanarak bir language model'i verimli bir şekilde fine-tune etmeyi gösterir.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def setup_lora_model(model_name="microsoft/DialoGPT-medium"):
    """
    LoRA konfigürasyonu ile modeli hazırlar
    """
    # Model ve tokenizer yükleme
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Pad token ayarlama
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA konfigürasyonu
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Low-rank dimension
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]  # Hangi katmanlar fine-tune edilecek
    )
    
    # LoRA modelini oluştur
    model = get_peft_model(model, lora_config)
    
    # Eğitilebilir parametreleri yazdır (fonksiyon None döndürür)
    model.print_trainable_parameters()
    
    # Hangi parametrelerin eğitildiğini göster
    print("\nEğitilen parametre listesi:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}: {param.shape}")
    
    print("\nDondurulmuş parametre örnekleri:")
    frozen_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if frozen_count < 3:  # İlk 3 örneği göster
                print(f"  ✗ {name}: {param.shape}")
                frozen_count += 1
            elif frozen_count == 3:
                print("  ... (ve diğer dondurulmuş parametreler)")
                break
    
    return model, tokenizer

def prepare_dataset(tokenizer, max_length=512):
    """
    Basit bir dataset oluşturur
    """
    # Örnek konuşma verileri
    conversations = [
        "Kullanıcı: Merhaba! Asistan: Merhaba! Size nasıl yardımcı olabilirim?",
        "Kullanıcı: Python öğrenmek istiyorum. Asistan: Harika! Python'a nereden başlamak istiyorsunuz?",
        "Kullanıcı: Makine öğrenmesi nedir? Asistan: Makine öğrenmesi, bilgisayarların verilerden öğrenmesini sağlayan bir tekniktir.",
    ]
    
    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Dataset oluştur
    dataset = Dataset.from_dict({"text": conversations})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def train_lora_model():
    """
    LoRA ile model eğitimi
    """
    print("LoRA model kurulumu...")
    model, tokenizer = setup_lora_model()
    
    print("Dataset hazırlığı...")
    train_dataset = prepare_dataset(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora_results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to=None,  # WandB/TensorBoard entegrasyonu kapalı
    )
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    print("Eğitim başlıyor...")
    trainer.train()
    
    # Modeli kaydet
    trainer.save_model("./lora_model")
    tokenizer.save_pretrained("./lora_model")
    
    print("LoRA eğitimi tamamlandı!")
    
    return model, tokenizer

def demonstrate_lora_benefits():
    """
    LoRA'nın faydalarını gösterir
    """
    print("\n=== LoRA Faydaları ===")
    print("1. Bellek Verimliliği: Sadece adapter katmanları eğitilir")
    print("2. Hızlı Eğitim: Daha az parametre = daha hızlı eğitim")
    print("3. Modülerlik: Farklı görevler için farklı adapter'lar")
    print("4. Düşük Disk Kullanımı: Sadece adapter ağırlıkları saklanır")
    
    # Pratik örnek
    model, tokenizer = setup_lora_model()
    
    # Original model size vs LoRA adapter size karşılaştırması
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nToplam parametreler: {total_params:,}")
    print(f"Eğitilebilir parametreler: {trainable_params:,}")
    print(f"Eğitilebilir oran: {100 * trainable_params / total_params:.2f}%")

if __name__ == "__main__":
    print("PEFT ve LoRA Demonstrasyonu")
    print("=" * 40)
    
    # LoRA faydalarını göster
    demonstrate_lora_benefits()
    
    # Eğitim yapmak için uncomment edin
    # model, tokenizer = train_lora_model()