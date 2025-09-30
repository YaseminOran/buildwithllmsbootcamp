"""
AutoTokenizer & AutoModel Yapısı + Pipeline ile Hızlı Model Çağırma

Bu modül AutoTokenizer ve AutoModel sınıflarının kullanımını ve 
pipeline ile hızlı model çağırma yöntemlerini gösterir.
"""

from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import time

def tokenizer_example(): 
    """AutoTokenizer kullanım örneği
    #AutoTokenizer kullanılarak bir metnin nasıl token'lara ayrıldığı
    # Encode (input_ids) ve attention mask üretimi
    # decode() ile input_ids → metne geri dönüş"""

    print("=== AutoTokenizer Örneği ===")
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "Hello, this is a sample text for tokenization."
    
    # Tokenization
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    # Encoding: token idlerinin tensor(sayısal vektör) karşılığıdır. 101 her metnin başına gelen özel tokendır.
    encoded = tokenizer.encode(text, return_tensors="pt")
    print(f"Encoded: {encoded}")
    
    # Encoding with attention mask
    encoded_dict = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(f"Input IDs: {encoded_dict['input_ids']}")
    print(f"Attention Mask: {encoded_dict['attention_mask']}") #modelin hangi tokenlara dikkat etmesi gerektiğini belirtir.
    # 1 bu tokena dikkat et 0 bu token padding, yok sayılabilir.

    # Decoding
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    print(f"Decoded: {decoded}") #düz metnimiz
    print()

def automodel_example():
    """AutoModel kullanım örneği
    AutoModel kullanarak bir transformer modelin çıktısına erişim
    .last_hidden_state üzerinden token embedding'lerin alınması
    İlk token ([CLS]) embedding'inin örneklenmesi
    """
    print("=== AutoModel Örneği ===")
    
    model_name = "bert-base-uncased" # bert modeli seçiliyor (küçük harfli)
    tokenizer = AutoTokenizer.from_pretrained(model_name) # metni tokenize edecek tokenizer yükleniyor
    model = AutoModel.from_pretrained(model_name) # model yükleniyor giriş olarak token id alır, çıktı olarak her token için hidden state verir.

    text = "This is an example sentence."
    
    # Tokenize : padding=True: uzunluklar eşitlenir (burada gerek yok, ama iyi alışkanlık) -truncation=True: metin uzunsa kesilir
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True) # metin pytorch tensörlwrine dönüşütürülüyor.

    
    # Model inference :Grad hesaplamasını kapatır, inference (tahmin) sırasında hız ve verimlilik sağlar.
    with torch.no_grad():
        outputs = model(**inputs) # Modelden çıktı alınır.

    
    # Extract embeddings
    last_hidden_states = outputs.last_hidden_state # bu tensörün şekli : (batch_size, sequence_length, hidden_size)

    print(f"Output shape: {last_hidden_states.shape}")  #Output shape: torch.Size([1, 8, 768]) her token için 768 boyutlu bir vektör elde edilir.
    print(f"CLS token embedding: {last_hidden_states[0][0][:5]}")  # İlk 5 değer
    print()

def pipeline_examples():
    """Pipeline ile hızlı model çağırma örnekleri
    pipeline() fonksiyonu ile sadece model adı vererek nasıl hızlı bir demo yapılır
    Her pipeline görev türü (sentiment-analysis, text-generation, question-answering, fill-mask) Hugging Face tarafından ön tanımlı olarak gelir
    Her görevin çıktısı, Hugging Face formatında dict objeleridir"""
    print("=== Pipeline Örnekleri ===")
    
    # Sentiment Analysis Pipeline: cümle olumlu mu olumsuz mu
    print("1. Sentiment Analysis:")
    sentiment_pipeline = pipeline("sentiment-analysis")
    text = "I love this product! It's amazing."
    result = sentiment_pipeline(text)
    print(f"Text: {text}")
    print(f"Result: {result}")
    print()
    
    # Text Generation Pipeline : gpt 2 ile verilen başlangıca metin üretir
    print("2. Text Generation:")
    text_generator = pipeline("text-generation", model="gpt2")
    prompt = "The future of artificial intelligence is"
    result = text_generator(prompt, max_length=50, num_return_sequences=1)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result[0]['generated_text']}")
    print()
    
    # Question Answering Pipeline : verilen bağlamdan sorunun cevabını bulur
    print("3. Question Answering:")
    qa_pipeline = pipeline("question-answering")
    context = "The capital of France is Paris. It is known for the Eiffel Tower."
    question = "What is the capital of France?"
    result = qa_pipeline(question=question, context=context)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} (score: {result['score']:.4f})")
    print()
    
    # Fill Mask Pipeline : [Mask] yerine en olası kelimeleri üretis
    print("4. Fill Mask:") 
    fill_mask = pipeline("fill-mask")
    mask_token = fill_mask.tokenizer.mask_token  # genellikle <mask> ya da [MASK]
    text = f"The food was{mask_token}."
    result = fill_mask(text)
    print(f"Text: {text}")
    print(f"Top predictions:")
    for i, pred in enumerate(result[:3]):
        print(f"  {i+1}. {pred['token_str']} (score: {pred['score']:.4f})")
    print()

def performance_comparison():
    """Manuel tokenization vs Pipeline performans karşılaştırması"""
    print("=== Performans Karşılaştırması ===")
    
    texts = [
        "This is a test sentence.",
        "Another example for testing.",
        "Performance comparison between methods.",
        "Pipeline vs manual approach.",
        "Speed and efficiency analysis."
    ]
    
    # Manuel approach
    start_time = time.time()
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
    
    manual_time = time.time() - start_time
    
    # Pipeline approach
    start_time = time.time()
    feature_extractor = pipeline("feature-extraction", model=model_name)
    
    for text in texts:
        features = feature_extractor(text)
    
    pipeline_time = time.time() - start_time
    
    print(f"Manuel yaklaşım süresi: {manual_time:.4f} saniye") #Her metin için tokenizer + model ayrı ayrı çalıştırılarak elde edilen toplam süre
    print(f"Pipeline yaklaşımı süresi: {pipeline_time:.4f} saniye") #Aynı metinler için pipeline("feature-extraction") ile elde edilen toplam süre
    print(f"Hız farkı: {manual_time/pipeline_time:.2f}x")
    print()

def custom_pipeline_example():
    """Özelleştirilmiş pipeline örneği
    Özelleştirilmiş bir modelin (distilbert-base-uncased-finetuned-sst-2-english) 
    pipeline’a manuel olarak bağlanması
    Modelin çıktılarının nasıl okunacağı (label, score)
    Farklı duygusal tonlardaki metinlere verilen cevapların yorumlanması"""
    print("=== Özelleştirilmiş Pipeline ===")
    
    # Specific model ile pipeline
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name
    )
    
    texts = [
        "I'm very happy today!",
        "This is terrible.",
        "The weather is okay.",
        "Amazing product, highly recommended!",
        "Not sure about this decision."
    ]
    
    print("Sentiment analysis sonuçları:")
    for text in texts:
        result = sentiment_pipeline(text)
        label = result[0]['label']
        score = result[0]['score']
        print(f"'{text}' -> {label} ({score:.4f})")
    print()

if __name__ == "__main__":
    print("AutoTokenizer & AutoModel + Pipeline Örnekleri\n")
    
    tokenizer_example()
    automodel_example()
    pipeline_examples()
    performance_comparison()
    custom_pipeline_example()
    
    print("Tüm örnekler tamamlandı!")