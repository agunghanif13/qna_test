import streamlit as st
from transformers import pipeline

# Load pipeline QA ekstraktif
qa_pipeline = pipeline(
    "text2text-generation",
    model="indolem/indobert-base-uncased",
    tokenizer="indolem/indobert-base-uncased"
)

# UI Streamlit
st.title("🤖 Tanya Jawab Bahasa Indonesia (IndoBERT)")
st.markdown("Masukkan konteks dan pertanyaan. Model akan mencoba memberikan jawaban berdasarkan konteks.")

# Input user
context = st.text_area("📚 Konteks:", height=200)
question = st.text_input("❓ Pertanyaan:")

# Proses saat tombol ditekan
if st.button("Jawab"):
    if not context.strip() or not question.strip():
        st.warning("Mohon masukkan konteks dan pertanyaan.")
    else:
        input_text = f"question: {question} context: {context}"
        output = qa_pipeline(input_text, max_length=200, do_sample=False)[0]["generated_text"]

        st.subheader("📝 Jawaban Naratif:")
        st.write(output)

