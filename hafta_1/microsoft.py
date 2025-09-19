import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline

#https://huggingface.co/settings/tokens

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")  # .env: HF_TOKEN=hf_...

MODEL_ID = "microsoft/DialoGPT-medium"

tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

gen = pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tok,
    device_map="auto",
    torch_dtype="auto", #torch_dtype=torch.float16
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    token=HF_TOKEN,      # kritik: pipeline’a da geçir
)

prompt = "Human: Hello! Bot: Hi! How can I help you? Human: Tell me about language models. Bot:"
print(gen(prompt)[0]["generated_text"])
