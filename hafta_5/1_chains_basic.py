"""
Hafta 5 - Bölüm 1: Temel Chain Yapıları
LangChain ile Chain oluşturma ve kullanma
"""

import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.schema import BaseOutputParser

# OpenAI API anahtarını yükle
from dotenv import load_dotenv
load_dotenv()

# LLM'i başlat
llm = OpenAI(temperature=0.7)

def basic_chain_example():
    """Temel LLMChain kullanımı"""
    print("=" * 50)
    print("1. TEMEL CHAIN KULLANIMI")
    print("=" * 50)
    
    # Prompt template oluştur
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Bu konu hakkında 3 cümlelik bir açıklama yaz: {topic}"
    )
    
    # Chain oluştur
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Chain'i çalıştır
    result = chain.run("Yapay zeka")
    print(f"Sonuç: {result}")
    return result

def sequential_chain_example():
    """Sıralı chain kullanımı - basit"""
    print("\n" + "=" * 50)
    print("2. SİRALI CHAIN KULLANIMI")
    print("=" * 50)
    
    # İlk chain: Hikaye başlangıcı
    first_prompt = PromptTemplate(
        input_variables=["theme"],
        template="Bu tema ile başlayan kısa bir hikaye başlangıcı yaz: {theme}"
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt)
    
    # İkinci chain: Hikaye sonu
    second_prompt = PromptTemplate(
        input_variables=["story_beginning"],
        template="Bu hikaye başlangıcını heyecanlı bir sonla tamamla:\n{story_beginning}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt)
    
    # Chainleri birleştir
    overall_chain = SimpleSequentialChain(
        chains=[first_chain, second_chain]
    )
    
    # Çalıştır
    result = overall_chain.run("uzay macerası")
    print(f"Tamamlanmış hikaye:\n{result}")
    return result

def complex_sequential_chain_example():
    """Karmaşık sıralı chain - çoklu girdi/çıktı"""
    print("\n" + "=" * 50)
    print("3. KARMAŞIK SİRALI CHAIN")
    print("=" * 50)
    
    # İlk chain: Ürün analizi
    product_analysis_prompt = PromptTemplate(
        input_variables=["product_name"],
        template="Bu ürünün özelliklerini ve hedef kitlesini analiz et: {product_name}"
    )
    product_analysis_chain = LLMChain(
        llm=llm, 
        prompt=product_analysis_prompt,
        output_key="analysis"
    )
    
    # İkinci chain: Pazarlama stratejisi
    marketing_prompt = PromptTemplate(
        input_variables=["product_name", "analysis"],
        template="""
        Ürün: {product_name}
        Analiz: {analysis}
        
        Bu analiz temelinde 3 pazarlama stratejisi öner:
        """
    )
    marketing_chain = LLMChain(
        llm=llm,
        prompt=marketing_prompt,
        output_key="marketing_strategy"
    )
    
    # Üçüncü chain: Bütçe önerisi
    budget_prompt = PromptTemplate(
        input_variables=["product_name", "marketing_strategy"],
        template="""
        Ürün: {product_name}
        Pazarlama Stratejisi: {marketing_strategy}
        
        Bu stratejiler için aylık bütçe dağılımı öner:
        """
    )
    budget_chain = LLMChain(
        llm=llm,
        prompt=budget_prompt,
        output_key="budget_plan"
    )
    
    # Tüm chainleri birleştir
    overall_chain = SequentialChain(
        chains=[product_analysis_chain, marketing_chain, budget_chain],
        input_variables=["product_name"],
        output_variables=["analysis", "marketing_strategy", "budget_plan"]
    )
    
    # Çalıştır
    result = overall_chain({"product_name": "Akıllı saat"})
    
    print("ÜRÜN ANALİZİ:")
    print(result["analysis"])
    print("\nPAZARLAMA STRATEJİSİ:")
    print(result["marketing_strategy"])
    print("\nBÜTÇE PLANI:")
    print(result["budget_plan"])
    
    return result

class JsonOutputParser(BaseOutputParser):
    """Özel output parser örneği"""
    
    def parse(self, text: str):
        """Metni JSON formatına çevirmeye çalış"""
        try:
            import json
            # Basit JSON parse denemesi
            lines = text.strip().split('\n')
            result = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip()] = value.strip()
            return result
        except:
            return {"raw_output": text}

def custom_output_parser_example():
    """Özel output parser kullanımı"""
    print("\n" + "=" * 50)
    print("4. ÖZEL OUTPUT PARSER")
    print("=" * 50)
    
    # Prompt template
    prompt = PromptTemplate(
        input_variables=["city"],
        template="""
        Bu şehir hakkında bilgileri şu formatta ver:
        Nüfus: [nüfus bilgisi]
        İklim: [iklim bilgisi]
        Meşhur yemek: [yemek bilgisi]
        
        Şehir: {city}
        """
    )
    
    # Parser ile chain
    parser = JsonOutputParser()
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=parser
    )
    
    result = chain.run("İstanbul")
    print("Parsed result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    print("LANGCHAIN CHAIN ÖRNEKLERİ")
    print("Bu örneklerde farklı chain türlerini öğreneceksiniz.\n")
    
    # Örnekleri çalıştır
    basic_chain_example()
    sequential_chain_example()
    complex_sequential_chain_example()
    custom_output_parser_example()
    
    print("\n" + "=" * 50)
    print("TÜM ÖRNEKLER TAMAMLANDI!")
    print("Chain'ler ile daha karmaşık iş akışları oluşturabilirsiniz.")
    print("=" * 50)