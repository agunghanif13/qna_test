import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import pandas as pd
from datetime import datetime
import re

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Tanya Jawab IndoBERT",
    page_icon="🤖",
    layout="wide"
)

# Cache untuk model
@st.cache_resource
def load_model():
    """Load model IndoBERT dan tokenizer"""
    try:
        model_name = "indolem/indobert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Buat pipeline untuk question answering
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return qa_pipeline, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def format_narrative_answer(question, answer, context, confidence):
    """Format jawaban menjadi naratif yang lebih natural"""
    
    # Template naratif berdasarkan jenis pertanyaan
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['apa', 'apakah']):
        if confidence > 0.8:
            narrative = f"Berdasarkan informasi yang tersedia, {answer}. "
        else:
            narrative = f"Kemungkinan besar, {answer}. "
    
    elif any(word in question_lower for word in ['siapa', 'who']):
        if confidence > 0.8:
            narrative = f"Berdasarkan konteks yang diberikan, yang dimaksud adalah {answer}. "
        else:
            narrative = f"Sepertinya yang dimaksud adalah {answer}. "
    
    elif any(word in question_lower for word in ['kapan', 'when']):
        if confidence > 0.8:
            narrative = f"Berdasarkan informasi yang ada, waktu yang dimaksud adalah {answer}. "
        else:
            narrative = f"Kemungkinan waktu yang dimaksud adalah {answer}. "
    
    elif any(word in question_lower for word in ['dimana', 'where']):
        if confidence > 0.8:
            narrative = f"Lokasi yang dimaksud berdasarkan konteks adalah {answer}. "
        else:
            narrative = f"Kemungkinan lokasi yang dimaksud adalah {answer}. "
    
    elif any(word in question_lower for word in ['mengapa', 'kenapa', 'why']):
        if confidence > 0.8:
            narrative = f"Alasan atau penjelasannya adalah {answer}. "
        else:
            narrative = f"Kemungkinan alasannya adalah {answer}. "
    
    elif any(word in question_lower for word in ['bagaimana', 'how']):
        if confidence > 0.8:
            narrative = f"Cara atau prosesnya adalah {answer}. "
        else:
            narrative = f"Kemungkinan cara atau prosesnya adalah {answer}. "
    
    else:
        if confidence > 0.8:
            narrative = f"Berdasarkan analisis terhadap konteks, jawabannya adalah {answer}. "
        else:
            narrative = f"Kemungkinan jawabannya adalah {answer}. "
    
    # Tambahkan informasi confidence jika rendah
    if confidence < 0.5:
        narrative += "Namun, tingkat kepercayaan jawaban ini masih rendah, sehingga mungkin perlu verifikasi lebih lanjut."
    elif confidence < 0.7:
        narrative += "Jawaban ini memiliki tingkat kepercayaan sedang."
    else:
        narrative += "Jawaban ini memiliki tingkat kepercayaan yang tinggi."
    
    return narrative

def main():
    st.title("🤖 Sistem Tanya Jawab dengan IndoBERT")
    st.markdown("---")
    
    # Load model
    with st.spinner("Memuat model IndoBERT..."):
        qa_pipeline, tokenizer = load_model()
    
    if qa_pipeline is None:
        st.error("Gagal memuat model. Pastikan koneksi internet stabil.")
        return
    
    st.success("Model IndoBERT berhasil dimuat!")
    
    # Sidebar untuk pengaturan
    st.sidebar.header("⚙️ Pengaturan")
    max_length = st.sidebar.slider("Panjang maksimal jawaban", 50, 500, 200)
    min_confidence = st.sidebar.slider("Ambang batas confidence", 0.0, 1.0, 0.1)
    
    # Area input
    st.header("📝 Input Pertanyaan dan Konteks")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        question = st.text_area(
            "Masukkan pertanyaan Anda:",
            placeholder="Contoh: Siapa presiden Indonesia?",
            height=100
        )
    
    with col2:
        context = st.text_area(
            "Masukkan konteks/teks referensi:",
            placeholder="Contoh: Joko Widodo adalah presiden Indonesia ke-7 yang menjabat sejak 2014...",
            height=100
        )
    
    # Contoh data untuk testing
    if st.button("📋 Gunakan Contoh Data"):
        st.session_state.example_question = "Siapa yang menjadi presiden Indonesia ke-7?"
        st.session_state.example_context = """
        Joko Widodo atau yang akrab disapa Jokowi adalah Presiden Republik Indonesia ke-7. 
        Ia menjabat sejak 20 Oktober 2014 setelah memenangkan Pilpres 2014. Sebelum menjadi presiden, 
        Jokowi pernah menjabat sebagai Walikota Solo (2005-2012) dan Gubernur DKI Jakarta (2012-2014). 
        Jokowi lahir di Solo pada 21 Juni 1961 dan merupakan pengusaha furniture sebelum terjun ke politik.
        """
        st.rerun()
    
    # Gunakan contoh data jika ada
    if 'example_question' in st.session_state:
        question = st.session_state.example_question
        context = st.session_state.example_context
        
        # Clear example data
        del st.session_state.example_question
        del st.session_state.example_context
    
    # Tombol untuk memproses
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        process_button = st.button("🔍 Proses Tanya Jawab", type="primary")
    
    # Proses tanya jawab
    if process_button:
        if not question or not context:
            st.error("Harap masukkan pertanyaan dan konteks!")
            return
        
        with st.spinner("Memproses pertanyaan..."):
            try:
                # Proses dengan model
                result = qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=max_length
                )
                
                answer = result['answer']
                confidence = result['score']
                start_pos = result['start']
                end_pos = result['end']
                
                # Format jawaban naratif
                narrative_answer = format_narrative_answer(question, answer, context, confidence)
                
                # Tampilkan hasil
                st.header("📊 Hasil Analisis")
                
                # Jawaban naratif
                st.subheader("💬 Jawaban Naratif")
                st.info(narrative_answer)
                
                # Detail teknis
                with st.expander("🔍 Detail Teknis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Confidence Score", f"{confidence:.3f}")
                        st.metric("Posisi Start", start_pos)
                        st.metric("Posisi End", end_pos)
                    
                    with col2:
                        st.text("Jawaban Mentah:")
                        st.code(answer)
                        
                        # Highlight jawaban dalam konteks
                        highlighted_context = context[:start_pos] + "**" + context[start_pos:end_pos] + "**" + context[end_pos:]
                        st.markdown("**Konteks dengan highlight:**")
                        st.markdown(highlighted_context)
                
                # Evaluasi kualitas jawaban
                st.subheader("📈 Evaluasi Kualitas")
                if confidence >= 0.8:
                    st.success("✅ Jawaban berkualitas tinggi")
                elif confidence >= 0.5:
                    st.warning("⚠️ Jawaban berkualitas sedang")
                else:
                    st.error("❌ Jawaban berkualitas rendah - perlu verifikasi")
                
                # Simpan ke riwayat
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'question': question,
                    'answer': answer,
                    'narrative': narrative_answer,
                    'confidence': confidence
                })
                
            except Exception as e:
                st.error(f"Error saat memproses: {str(e)}")
    
    # Riwayat tanya jawab
    if 'history' in st.session_state and st.session_state.history:
        st.header("📜 Riwayat Tanya Jawab")
        
        # Tombol untuk clear history
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()
        
        # Tampilkan riwayat
        for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Tampilkan 5 terakhir
            with st.expander(f"Q{len(st.session_state.history)-i}: {item['question'][:50]}..."):
                st.write(f"**Waktu:** {item['timestamp']}")
                st.write(f"**Pertanyaan:** {item['question']}")
                st.write(f"**Jawaban Naratif:** {item['narrative']}")
                st.write(f"**Confidence:** {item['confidence']:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Dibuat dengan ❤️ menggunakan Streamlit dan IndoBERT</p>
            <p><small>Model: indolem/indobert-base-uncased</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()