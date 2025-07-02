from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer dan model
model_name = "cahya/gpt2-large-indonesian-522M"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",             # Distribusi otomatis ke CPU/GPU
    torch_dtype="auto",            # Otomatis gunakan precision yang sesuai
    trust_remote_code=True
)

def tanya_gpt2(pertanyaan):
    prompt = f"Pertanyaan: {pertanyaan}\nJawaban:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Hapus bagian prompt dari output
    jawaban = output[len(prompt):].strip()
    return jawaban


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
print("🤖 Halo! Saya adalah asisten AI berbasis Qwen2.5-3B-Instruct. Silakan ajukan pertanyaan (ketik 'keluar' untuk berhenti).")
while True:
    user_input = input("🧑 Anda: ")
    if user_input.lower() in ["keluar", "exit", "quit"]:
        print("👋 Sampai jumpa!")
        break
    jawaban = tanya_gpt2(user_input)
    print(f"🧠 GPT2Indo: {jawaban}")