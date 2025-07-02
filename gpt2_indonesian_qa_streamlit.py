import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import pandas as pd
from datetime import datetime
import re
import random

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Tanya Jawab GPT2 Indonesia",
    page_icon="🤖",
    layout="wide"
)

# Cache untuk model
@st.cache_resource
def load_model():
    """Load model GPT2 Indonesia dan tokenizer"""
    try:
        model_name = "cahya/gpt2-large-indonesian-522M"
        
        with st.spinner("Mengunduh model GPT2 Indonesia (522M)... Ini mungkin memakan waktu beberapa menit untuk pertama kali."):
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Buat pipeline untuk text generation
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        return generator, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_qa_prompt(question, context=""):
    """Membuat prompt untuk tanya jawab"""
    
    # Template prompt yang berbeda berdasarkan jenis pertanyaan
    question_lower = question.lower()
    
    if context:
        if any(word in question_lower for word in ['apa', 'apakah']):
            prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban: Berdasarkan konteks di atas,"
        elif any(word in question_lower for word in ['siapa', 'who']):
            prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban: Menurut informasi yang diberikan,"
        elif any(word in question_lower for word in ['kapan', 'when']):
            prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban: Berdasarkan waktu yang disebutkan,"
        elif any(word in question_lower for word in ['dimana', 'where']):
            prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban: Lokasi yang dimaksud adalah"
        elif any(word in question_lower for word in ['mengapa', 'kenapa', 'why']):
            prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban: Alasannya adalah karena"
        elif any(word in question_lower for word in ['bagaimana', 'how']):
            prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban: Cara atau prosesnya adalah"
        else:
            prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban:"
    else:
        # Tanpa konteks, gunakan pengetahuan umum
        prompt = f"Pertanyaan: {question}\nJawaban:"
    
    return prompt

def clean_generated_text(text, original_prompt):
    """Membersihkan teks yang dihasilkan"""
    
    # Hapus prompt dari hasil
    if original_prompt in text:
        text = text.replace(original_prompt, "").strip()
    
    # Cari jawaban setelah "Jawaban:"
    if "Jawaban:" in text:
        text = text.split("Jawaban:")[-1].strip()
    
    # Bersihkan karakter yang tidak diinginkan
    text = re.sub(r'\n+', ' ', text)  # Ganti multiple newlines dengan space
    text = re.sub(r'\s+', ' ', text)  # Ganti multiple spaces dengan single space
    text = text.strip()
    
    # Potong di titik yang wajar jika terlalu panjang
    sentences = text.split('.')
    if len(sentences) > 3:  # Batasi maksimal 3 kalimat
        text = '. '.join(sentences[:3]) + '.'
    elif not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
        text += '.'
    
    return text

def generate_answer(generator, question, context="", max_length=150, temperature=0.7, top_p=0.9):
    """Generate jawaban menggunakan GPT2"""
    
    try:
        # Buat prompt
        prompt = create_qa_prompt(question, context)
        
        # Generate jawaban
        result = generator(
            prompt,
            max_length=len(prompt.split()) + max_length,
            min_length=len(prompt.split()) + 20,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
        
        generated_text = result[0]['generated_text']
        answer = clean_generated_text(generated_text, prompt)
        
        return answer, True
    
    except Exception as e:
        return f"Error generating answer: {str(e)}", False

def main():
    st.title("🤖 Sistem Tanya Jawab dengan GPT2 Indonesia")
    st.markdown("### Model: cahya/gpt2-large-indonesian-522M")
    st.markdown("---")
    
    # Load model
    with st.spinner("Memuat model GPT2 Indonesia..."):
        generator, tokenizer = load_model()
    
    if generator is None:
        st.error("Gagal memuat model. Pastikan koneksi internet stabil dan coba lagi.")
        return
    
    st.success("✅ Model GPT2 Indonesia berhasil dimuat!")
    
    # Sidebar untuk pengaturan
    st.sidebar.header("⚙️ Pengaturan Generation")
    max_length = st.sidebar.slider("Panjang maksimal jawaban", 50, 300, 150)
    temperature = st.sidebar.slider("Temperature (kreativitas)", 0.1, 1.0, 0.7, 0.1)
    top_p = st.sidebar.slider("Top-p (keragaman)", 0.1, 1.0, 0.9, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Tips:**")
    st.sidebar.markdown("- Temperature tinggi = jawaban lebih kreatif")
    st.sidebar.markdown("- Temperature rendah = jawaban lebih konservatif")
    st.sidebar.markdown("- Top-p tinggi = lebih beragam")
    
    # Area input
    st.header("📝 Input Pertanyaan")
    
    # Mode selection
    mode = st.radio(
        "Pilih mode tanya jawab:",
        ["Dengan Konteks", "Pengetahuan Umum"],
        horizontal=True
    )
    
    question = st.text_area(
        "Masukkan pertanyaan Anda:",
        placeholder="Contoh: Siapa presiden Indonesia saat ini?",
        height=100
    )
    
    context = ""
    if mode == "Dengan Konteks":
        context = st.text_area(
            "Masukkan konteks/teks referensi:",
            placeholder="Masukkan teks yang berisi informasi untuk menjawab pertanyaan...",
            height=150
        )
    
    # Contoh data
    st.markdown("---")
    st.subheader("📋 Contoh Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏛️ Sejarah Indonesia"):
            st.session_state.example_question = "Kapan Indonesia merdeka dan siapa yang memproklamasikannya?"
            st.session_state.example_context = """
            Indonesia merdeka pada tanggal 17 Agustus 1945. Proklamasi kemerdekaan Indonesia dibacakan oleh 
            Ir. Soekarno (yang kemudian menjadi presiden pertama) dan didampingi oleh Mohammad Hatta 
            (yang menjadi wakil presiden pertama). Proklamasi tersebut dibacakan di Jalan Pegangsaan Timur 56, Jakarta.
            """ if mode == "Dengan Konteks" else ""
    
    with col2:
        if st.button("🌍 Geografi"):
            st.session_state.example_question = "Apa ibu kota Indonesia dan dimana letaknya?"
            st.session_state.example_context = """
            Jakarta adalah ibu kota negara Indonesia yang terletak di pulau Jawa bagian barat. 
            Jakarta merupakan kota terbesar di Indonesia dengan populasi lebih dari 10 juta jiwa. 
            Kota ini juga merupakan pusat pemerintahan, bisnis, dan ekonomi Indonesia.
            """ if mode == "Dengan Konteks" else ""
    
    with col3:
        if st.button("🎨 Budaya"):
            st.session_state.example_question = "Apa saja makanan tradisional Indonesia yang terkenal?"
            st.session_state.example_context = """
            Indonesia memiliki berbagai makanan tradisional yang terkenal seperti nasi gudeg dari Yogyakarta, 
            rendang dari Sumatera Barat, sate dari Jawa, gado-gado dari Jakarta, dan pempek dari Palembang. 
            Setiap daerah memiliki cita rasa dan cara memasak yang unik.
            """ if mode == "Dengan Konteks" else ""
    
    # Gunakan contoh data jika ada
    if 'example_question' in st.session_state:
        question = st.session_state.example_question
        if mode == "Dengan Konteks" and 'example_context' in st.session_state:
            context = st.session_state.example_context
        
        # Clear example data
        if 'example_question' in st.session_state:
            del st.session_state.example_question
        if 'example_context' in st.session_state:
            del st.session_state.example_context
        st.rerun()
    
    # Tombol untuk memproses
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        process_button = st.button("🔍 Generate Jawaban", type="primary")
    
    # Proses tanya jawab
    if process_button:
        if not question:
            st.error("Harap masukkan pertanyaan!")
            return
        
        if mode == "Dengan Konteks" and not context:
            st.error("Harap masukkan konteks untuk mode ini!")
            return
        
        with st.spinner("Menghasilkan jawaban naratif..."):
            # Generate jawaban
            answer, success = generate_answer(
                generator, 
                question, 
                context if mode == "Dengan Konteks" else "",
                max_length, 
                temperature, 
                top_p
            )
            
            if success:
                # Tampilkan hasil
                st.header("💬 Jawaban yang Dihasilkan")
                
                # Jawaban dalam box yang menarik
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f0f8ff;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid #4CAF50;
                        margin: 10px 0;
                    ">
                        <h4 style="color: #2E7D32; margin-top: 0;">📝 Jawaban Naratif:</h4>
                        <p style="font-size: 16px; line-height: 1.6; margin-bottom: 0;">{answer}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Detail informasi
                with st.expander("🔍 Detail Generation"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Panjang Jawaban", f"{len(answer.split())} kata")
                        st.metric("Temperature", temperature)
                        st.metric("Top-p", top_p)
                    
                    with col2:
                        st.text("Mode:")
                        st.code(mode)
                        st.text("Prompt yang digunakan:")
                        prompt_preview = create_qa_prompt(question, context if mode == "Dengan Konteks" else "")
                        st.code(prompt_preview[:200] + "..." if len(prompt_preview) > 200 else prompt_preview)
                
                # Simpan ke riwayat
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'question': question,
                    'answer': answer,
                    'mode': mode,
                    'context': context if mode == "Dengan Konteks" else "Tidak ada",
                    'temperature': temperature,
                    'top_p': top_p
                })
                
            else:
                st.error(f"Gagal menghasilkan jawaban: {answer}")
    
    # Riwayat tanya jawab
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("---")
        st.header("📜 Riwayat Tanya Jawab")
        
        # Tombol untuk clear history
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Hapus Riwayat"):
                st.session_state.history = []
                st.rerun()
        
        # Tampilkan riwayat (5 terakhir)
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Q{len(st.session_state.history)-i}: {item['question'][:60]}..."):
                st.write(f"**⏰ Waktu:** {item['timestamp']}")
                st.write(f"**❓ Pertanyaan:** {item['question']}")
                st.write(f"**💡 Jawaban:** {item['answer']}")
                st.write(f"**🔧 Mode:** {item['mode']}")
                if item['context'] != "Tidak ada":
                    st.write(f"**📖 Konteks:** {item['context'][:100]}...")
                st.write(f"**🌡️ Settings:** Temp={item['temperature']}, Top-p={item['top_p']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 Dibuat dengan ❤️ menggunakan Streamlit dan GPT2 Indonesia</p>
            <p><small>Model: cahya/gpt2-large-indonesian-522M (522M parameters)</small></p>
            <p><small>💡 Tip: Gunakan pertanyaan yang jelas dan spesifik untuk hasil terbaik</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()