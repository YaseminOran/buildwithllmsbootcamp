from transformers import AutoTokenizer, pipeline

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generator = pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=250,
    temperature=0.7
)

prompt = "LLM nedir? Kısaca cevapla"

response = generator(prompt)
print(response[0]["generated_text"])