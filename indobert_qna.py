from transformers import pipeline

# Muat model QA IndoBERT
qa_pipeline = pipeline(
    "question-answering",
    model="indolem/indobert-base-uncased",
    tokenizer="indolem/indobert-base-uncased"
)

def tanya_jawab(model, konteks):
    print("🤖 Halo! Saya adalah asisten AI berbasis model indolem/indobert-base-uncased.")
    print("Silakan ajukan pertanyaan berdasarkan konteks di bawah ini:")
    print("\n📜 Konteks:\n", konteks)
    print("\nKetik 'keluar' untuk berhenti.\n")

    while True:
        pertanyaan = input("🧑 Anda: ")
        if pertanyaan.lower() in ["keluar", "exit", "quit"]:
            print("👋 Sampai jumpa!")
            break
        
        hasil = model(question=pertanyaan, context=konteks)
        jawaban = hasil["answer"]
        skor = hasil["score"]

        print(f"🧠 Qwen: Jawaban: '{jawaban}' (Skor keyakinan: {skor:.4f})")

# Contoh konteks
konteks_default = """
Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara dan Oseania.
Ibukota Indonesia adalah Jakarta. Bahasa resmi Indonesia adalah Bahasa Indonesia.
Negara ini memiliki lebih dari 17.000 pulau dan lebih dari 300 kelompok etnis.
"""

# Jalankan chatbot
tanya_jawab(qa_pipeline, konteks_default)