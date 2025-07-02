from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer dan model
model_name = "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",             # Distribusi otomatis ke CPU/GPU
    torch_dtype="auto",            # Otomatis gunakan precision yang sesuai
    trust_remote_code=True
)

# Fungsi tanya jawab
def tanya_qwen(pertanyaan):
    # Format pesan dengan system prompt (opsional)
    messages = [
        {"role": "system", "content": "Anda adalah asisten AI yang membantu."},
        {"role": "user", "content": pertanyaan}
    ]
    
    # Gunakan tokenizer untuk menyusun input
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate jawaban
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,         # Maksimal panjang jawaban
        do_sample=True,             # Aktifkan sampling untuk respons lebih alami
        temperature=0.7,            # Kontrol kreativitas
        top_p=0.95                  # Nucleus sampling
    )

    # Decode hasil
    output = tokenizer.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]
    return output

# Loop interaktif
print("🤖 Halo! Saya adalah asisten AI berbasis sahabat_ai_gemmav1. Silakan ajukan pertanyaan (ketik 'keluar' untuk berhenti).")
while True:
    user_input = input("🧑 Anda: ")
    if user_input.lower() in ["keluar", "exit", "quit"]:
        print("👋 Sampai jumpa!")
        break
    jawaban = tanya_qwen(user_input)
    print(f"🧠 Qwen: {jawaban}")
    print(f"⏱️ Durasi inferensi: {waktu:.4f} detik ({waktu * 1000:.2f} ms)\n")