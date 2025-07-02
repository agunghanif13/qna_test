from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Muat tokenizer dan model
model_name = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"

print("🔄 Memuat tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("🔄 Memuat model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,        # Hemat memori
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()

def generate_response(question):
    start_time = time.time()  # Mulai timer

    # Gunakan chat template jika tersedia
    messages = [
        {"role": "system", "content": "Anda adalah asisten AI Sahabat AI yang ramah dan membantu."},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    end_time = time.time()  # Akhiri timer

    # Hilangkan bagian system prompt dan user input
    answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    answer = answer.split("<|eot_id|>")[0].strip()

    duration = end_time - start_time  # Hitung durasi
    return answer, duration

# Loop interaktif
print("\n🤖 Halo! Saya adalah asisten AI berbasis Llama3-8B-SahabatAI.")
print("Silakan ajukan pertanyaan (ketik 'keluar' untuk berhenti).\n")

while True:
    user_input = input("🧑 Anda: ")
    if user_input.lower() in ["keluar", "exit", "quit"]:
        print("👋 Sampai jumpa!")
        break
    
    jawaban, waktu = generate_response(user_input)
    print(f"🧠 Bot: {jawaban}")
    print(f"⏱️ Durasi inferensi: {waktu:.4f} detik ({waktu * 1000:.2f} ms)\n")