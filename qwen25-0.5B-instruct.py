from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Login ke Hugging Face (opsional jika model memerlukan autentikasi)
from huggingface_hub import login
login(token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")  # Ganti dengan token kamu

# Muat tokenizer dan model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("🔄 Memuat tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("🔄 Memuat model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,      # Atau gunakan torch.bfloat16 jika tersedia
    trust_remote_code=True
).eval()

def generate_response(question):
    start_time = time.time()  # Mulai timer

    prompt = f"<|begin_of_sentence|>{question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()  # Akhiri timer

    # Hitung durasi
    duration = end_time - start_time
    return response.strip(), duration

# Loop interaktif
print("\n🤖 Halo! Saya adalah asisten AI berbasis Qwen3-1.5B-Instruct.")
print("Silakan ajukan pertanyaan (ketik 'keluar' untuk berhenti).\n")

while True:
    user_input = input("🧑 Anda: ")
    if user_input.lower() in ["keluar", "exit", "quit"]:
        print("👋 Sampai jumpa!")
        break
    
    jawaban, waktu = generate_response(user_input)
    print(f"🧠 Bot: {jawaban}")
    print(f"⏱️ Durasi inferensi: {waktu:.4f} detik ({waktu * 1000:.2f} ms)\n")