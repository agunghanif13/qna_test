import streamlit as st
from transformers import pipeline

# Load pipeline model generatif QA
qa_pipeline = pipeline(
    "text2text-generation",
    model="cahya/bert-base-indonesian-tydiqa",
    tokenizer="cahya/bert-base-indonesian-tydiqa"
)

st.title("📘 QA Bahasa Indonesia dengan Jawaban Panjang (BERT)")
st.markdown("Masukkan teks konteks dan pertanyaan, model akan memberikan jawaban naratif.")

# Input konteks dan pertanyaan
context = st.text_area("📝 Konteks:", height=200)
question = st.text_input("❓ Pertanyaan:")

if st.button("Jawab"):
    if not context.strip() or not question.strip():
        st.warning("Mohon masukkan konteks dan pertanyaan.")
    else:
        # Format input untuk T5
        input_text = f"question: {question} context: {context}"
        result = qa_pipeline(input_text, max_length=150, do_sample=False)[0]["generated_text"]
        st.subheader("Jawaban:")
        st.write(result)