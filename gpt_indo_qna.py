from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Muat tokenizer dan model
model_name = "cahya/gpt2-large-indonesian-522M"

print("🔄 Memuat model dan tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Jika ada GPU, gunakan CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model dimuat menggunakan device: {device}")

def generate_response(question, max_length=100):
    # Format input
    prompt = f"Pertanyaan: {question}\nJawaban:"
    
    # Tokenisasi input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate jawaban
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    # Decode hasil
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Ambil hanya bagian jawaban
    answer = response.split("Jawaban:")[-1].strip()
    return answer

# Loop interaktif
print("\n🤖 Halo! Saya adalah asisten AI berbasis model cahya/gpt2-large-indonesian-522M.")
print("Silakan ajukan pertanyaan (ketik 'keluar' untuk berhenti).\n")

while True:
    user_input = input("🧑 Anda: ")
    if user_input.lower() in ["keluar", "exit", "quit"]:
        print("👋 Sampai jumpa!")
        break
    jawaban = generate_response(user_input)
    print(f"🧠 Bot: {jawaban}")