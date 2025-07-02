from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Login ke Hugging Face (opsional jika model memerlukan autentikasi)
from huggingface_hub import login
login(token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")  # Ganti dengan token kamu

# Muat tokenizer dan model
model_name = "Qwen/Qwen2.5-3B-Instruct"

print("🔄 Memuat tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("🔄 Memuat model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",             # Distribusi otomatis ke GPU/CPU
    torch_dtype=torch.float16,     # Menghemat memori
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()

# Fungsi generate jawaban
def generate_response(question):
    prompt = f"<|begin_of_sentence|>{question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Loop interaktif
print("\n🤖 Halo! Saya adalah asisten AI berbasis Qwen2.5-3B-Instruct.")
print("Silakan ajukan pertanyaan (ketik 'keluar' untuk berhenti).\n")

while True:
    user_input = input("🧑 Anda: ")
    if user_input.lower() in ["keluar", "exit", "quit"]:
        print("👋 Sampai jumpa!")
        break
    jawaban = generate_response(user_input)
    print(f"🧠 Bot: {jawaban}")