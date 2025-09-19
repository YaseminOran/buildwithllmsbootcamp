# Hafta 2: Prompt Engineering ve API Tabanlı Kullanım

Bu modülde prompt engineering teknikleri ve OpenAI API kullanımı örnekleri bulunmaktadır.

## Kurulum

1. Sanal ortamı aktifleştirin:
```bash
source prompt/bin/activate  # Mac/Linux
# veya
prompt\Scripts\activate     # Windows
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. `.env` dosyasında OpenAI API anahtarınızı ayarlayın:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Modüller

### 1. Zero-Shot Prompting (`01_zero_shot.py`)
- Örnek vermeden doğrudan görev tanımı ile prompt yazma
- Duygusal analiz, çeviri, özet çıkarma örnekleri

### 2. Few-Shot Prompting (`02_few_shot.py`) 
- Modelin öğrenmesi için birkaç örnek vererek prompt yazma
- Sınıflandırma, varlık çıkarma, metin formatlama örnekleri

### 3. Chain of Thought (`03_chain_of_thought.py`)
- Modelin adım adım düşünmesini sağlama
- Matematik problemleri, mantıksal akıl yürütme, problem çözme

### 4. Role-Based Prompting (`04_role_based.py`)
- Modele belirli rol vererek spesifik yanıtlar alma
- Pazarlama uzmanı, teknik yazar, mali müşavir rolleri

### 5. ChatCompletion API (`05_chatcompletion_api.py`)
- OpenAI ChatCompletion API detaylı kullanımı
- Farklı parametreler, streaming, konuşma yönetimi

### 6. Function Calling (`06_function_calling.py`)
- Function calling temel chatbot sınıfı
- Geometrik alan hesaplama (dikdörtgen, daire, üçgen)
- Hava durumu, döviz çevirme, zaman bilgisi, e-posta doğrulama
- Temel chatbot mimarisi ve conversation management

### 7. Akıllı Chatbot (`07_chatbot_with_functions.py`)
- 06'dan inherit eden genişletilmiş chatbot sistemi  
- Temel fonksiyonlar (06'dan): alan hesaplama, hava durumu, döviz, zaman, e-posta
- Yeni fonksiyonlar: web araması, hatırlatıcı, yapılacaklar listesi, rastgele bilgiler
- Code reusability ve inheritance örneği

### 8. Basit Chatbot (`08_simple_chatbot.py`)
- Basit function calling chatbot sınıfı
- Hesap makinesi (4 işlem) ve not yönetimi (kaydet/listele)
- Available functions dictionary kullanımı
- Anlaşılması kolay yapı

### 9. Web Chatbot (`09_web_chatbot.py`)
- Flask ile web arayüzlü chatbot
- Gerçek zamanlı döviz kurları (exchangerate-api.com)
- Hesaplama, not alma, bilgi arama, not listeleme
- Modern web arayüzü ve görsel istatistikler

## Çalıştırma

Her modülü ayrı ayrı çalıştırabilirsiniz:

```bash
python 01_zero_shot.py
python 02_few_shot.py
python 03_chain_of_thought.py
python 04_role_based.py
python 05_chatcompletion_api.py
python 06_function_calling.py
python 07_chatbot_with_functions.py
python 08_simple_chatbot.py
python 09_web_chatbot.py  # Web tarayıcıda http://localhost:5000
```

## Dikkat Edilecek Noktalar

- **API Güvenliği**: API anahtarınızı `.env` dosyasında güvenli şekilde saklayın
- **Token Takibi**: Token kullanımınızı takip edin (özellikle GPT-4 ile)
- **Rate Limiting**: API çağırım limitlerini göz önünde bulundurun
- **Error Handling**: Fonksiyon çağırımlarında hata yönetimini ihmal etmeyin
- **Code Reusability**: 07'de gösterildiği gibi inheritance kullanarak kod tekrarından kaçının
- **Function Definitions**: OpenAI format'ına uygun fonksiyon tanımları yapın

## Öğrenme Sırası

1. **01-04**: Temel prompt teknikleri
2. **05**: ChatCompletion API özellikleri  
3. **06**: Function calling temel mimarisi ⭐
4. **07**: Inheritance ile kod genişletme ⭐
5. **08**: Basit implementation
6. **09**: Web arayüzü entegrasyonu

⭐ **Ana odak noktaları**: 06 ve 07 arasındaki inheritance ilişkisini anlamak