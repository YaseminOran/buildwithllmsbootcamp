"""
GPT, BERT ve T5 Modellerinin Farkları ve Pipeline Entegrasyonu

Bu modül GPT, BERT ve T5 modellerinin temel farklarını gösterir ve
pipeline entegrasyonu ile üç modeli tek satırda test eder.
"""

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import time

def model_architectures_overview():
    """Model mimarilerinin genel açıklaması"""
    print("=== Model Mimarileri Genel Bakış ===")
    print("GPT (Generative Pre-trained Transformer):")
    print("  - Decoder-only mimari")
    print("  - Autoregressive (left-to-right) text generation")
    print("  - Causal attention (sadece önceki tokenlara bakar)")
    print("  - Kullanım: Text generation, completion")
    print()
    
    print("BERT (Bidirectional Encoder Representations from Transformers):")
    print("  - Encoder-only mimari")
    print("  - Bidirectional attention (her iki yöne bakar)")
    print("  - Masked Language Modeling ile eğitilir")
    print("  - Kullanım: Classification, NER, QA, feature extraction")
    print()
    
    print("T5 (Text-to-Text Transfer Transformer):")
    print("  - Encoder-decoder mimari")
    print("  - Her task text-to-text formatında")
    print("  - Prefix ile task belirtilir")
    print("  - Kullanım: Translation, summarization, QA, generation")
    print()

def test_gpt_models():
    """GPT modellerini test et"""
    print("=== GPT Modelleri Test ===")
    
    # Text Generation
    print("1. Text Generation (GPT-2):")
    generator = pipeline("text-generation", model="gpt2")
    prompt = "The benefits of machine learning include"
    result = generator(prompt, max_length=80, num_return_sequences=1, 
                      temperature=0.7, do_sample=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result[0]['generated_text']}")
    print()
    
    # Conversation
    print("2. Conversational AI:")
    conversational = pipeline("conversational", model="microsoft/DialoGPT-medium")
    from transformers import Conversation
    
    conversation = Conversation("Hello, how are you?")
    result = conversational(conversation)
    print(f"User: Hello, how are you?")
    print(f"Bot: {result.messages[-1]['content']}")
    print()

def test_bert_models():
    """BERT modellerini test et"""
    print("=== BERT Modelleri Test ===")
    
    # Sentiment Analysis
    print("1. Sentiment Analysis:")
    sentiment = pipeline("sentiment-analysis", 
                        model="nlptown/bert-base-multilingual-uncased-sentiment")
    text = "This product is absolutely amazing!"
    result = sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']} (score: {result[0]['score']:.4f})")
    print()
    
    # Fill Mask
    print("2. Fill Mask:")
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    text = "The capital of [MASK] is Paris."
    result = fill_mask(text)
    print(f"Text: {text}")
    print("Top predictions:")
    for i, pred in enumerate(result[:3]):
        print(f"  {i+1}. {pred['token_str']} (score: {pred['score']:.4f})")
    print()
    
    # Question Answering
    print("3. Question Answering:")
    qa = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    context = "BERT is a transformer model developed by Google. It uses bidirectional attention."
    question = "Who developed BERT?"
    result = qa(question=question, context=context)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} (score: {result['score']:.4f})")
    print()
    
    # Named Entity Recognition
    print("4. Named Entity Recognition:")
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                   aggregation_strategy="simple")
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    result = ner(text)
    print(f"Text: {text}")
    print("Entities:")
    for entity in result:
        print(f"  {entity['word']}: {entity['entity_group']} (score: {entity['score']:.4f})")
    print()

def test_t5_models():
    """T5 modellerini test et"""
    print("=== T5 Modelleri Test ===")
    
    # Translation
    print("1. Translation:")
    translator = pipeline("translation_en_to_fr", model="t5-small")
    text = "Hello, how are you today?"
    result = translator(text)
    print(f"English: {text}")
    print(f"French: {result[0]['translation_text']}")
    print()
    
    # Summarization
    print("2. Summarization:")
    summarizer = pipeline("summarization", model="t5-small")
    text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. It has applications in many fields including healthcare, 
    finance, and technology. Deep learning, a subset of machine learning, uses neural 
    networks with multiple layers to solve complex problems. These models have achieved 
    remarkable success in image recognition, natural language processing, and game playing.
    """
    result = summarizer(text, max_length=50, min_length=20, do_sample=False)
    print(f"Original text: {text.strip()}")
    print(f"Summary: {result[0]['summary_text']}")
    print()
    
    # Question Answering (T5 style)
    print("3. Question Answering (T5):")
    qa_t5 = pipeline("text2text-generation", model="t5-small")
    context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
    question = "question: Where is the Eiffel Tower located? context: " + context
    result = qa_t5(question, max_length=20)
    print(f"Question: Where is the Eiffel Tower located?")
    print(f"Context: {context}")
    print(f"Answer: {result[0]['generated_text']}")
    print()

def single_line_comparison():
    """Tek satırda üç modeli test etme"""
    print("=== Tek Satırda Model Karşılaştırması ===")
    
    # Aynı task için farklı modeller
    text = "I love this new technology!"
    
    print(f"Test text: {text}")
    print()
    
    # Sentiment analysis with different models
    models = [
        ("BERT", "nlptown/bert-base-multilingual-uncased-sentiment"),
        ("RoBERTa", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
        ("DistilBERT", "distilbert-base-uncased-finetuned-sst-2-english")
    ]
    
    print("Sentiment Analysis karşılaştırması:")
    for model_name, model_path in models:
        try:
            classifier = pipeline("sentiment-analysis", model=model_path)
            result = classifier(text)[0]
            print(f"{model_name:12}: {result['label']} (score: {result['score']:.4f})")
        except Exception as e:
            print(f"{model_name:12}: Error - {str(e)[:50]}")
    print()

def performance_comparison():
    """Model performanslarını karşılaştır"""
    print("=== Performans Karşılaştırması ===")
    
    text = "This is a sample text for performance testing."
    
    models_to_test = [
        ("BERT-base", "bert-base-uncased", "feature-extraction"),
        ("DistilBERT", "distilbert-base-uncased", "feature-extraction"),
        ("GPT-2", "gpt2", "text-generation")
    ]
    
    for model_name, model_path, task in models_to_test:
        try:
            start_time = time.time()
            
            if task == "text-generation":
                pipe = pipeline(task, model=model_path)
                result = pipe(text, max_length=len(text.split()) + 10, 
                             num_return_sequences=1, do_sample=False)
            else:
                pipe = pipeline(task, model=model_path)
                result = pipe(text)
            
            end_time = time.time()
            
            print(f"{model_name:12}: {end_time - start_time:.4f} saniye")
            
        except Exception as e:
            print(f"{model_name:12}: Error - {str(e)[:30]}")
    
    print()

def model_size_comparison():
    """Model boyutlarını karşılaştır"""
    print("=== Model Boyut Karşılaştırması ===")
    
    models = [
        ("BERT-base", "bert-base-uncased"),
        ("DistilBERT", "distilbert-base-uncased"),
        ("GPT-2", "gpt2"),
        ("T5-small", "t5-small")
    ]
    
    for model_name, model_path in models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            
            # Parameter sayısını hesapla
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"{model_name:12}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Vocab size: {tokenizer.vocab_size:,}")
            print()
            
        except Exception as e:
            print(f"{model_name:12}: Error loading model")

def use_case_recommendations():
    """Model kullanım önerileri"""
    print("=== Model Kullanım Önerileri ===")
    
    recommendations = {
        "GPT": [
            "✅ Text generation",
            "✅ Creative writing",
            "✅ Code completion",
            "✅ Conversational AI",
            "❌ Text classification",
            "❌ Named Entity Recognition"
        ],
        "BERT": [
            "✅ Text classification",
            "✅ Sentiment analysis",
            "✅ Question answering",
            "✅ Named Entity Recognition",
            "✅ Feature extraction",
            "❌ Text generation"
        ],
        "T5": [
            "✅ Translation",
            "✅ Summarization",
            "✅ Question answering",
            "✅ Text-to-text tasks",
            "✅ Data-to-text generation",
            "⚠️  Requires task-specific prompts"
        ]
    }
    
    for model, uses in recommendations.items():
        print(f"{model} Model:")
        for use in uses:
            print(f"  {use}")
        print()

if __name__ == "__main__":
    print("GPT, BERT ve T5 Model Karşılaştırması\n")
    
    model_architectures_overview()
    test_bert_models()
    test_gpt_models()
    test_t5_models()
    single_line_comparison()
    performance_comparison()
    model_size_comparison()
    use_case_recommendations()
    
    print("Model karşılaştırması tamamlandı!")