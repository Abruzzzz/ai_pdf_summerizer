import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Title
st.set_page_config(page_title="SmartNotes AI", layout="centered")
st.title("🧠 SmartNotes AI")
st.subheader("Upload a PDF and get an AI-powered summary!")

# Load summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# File upload
uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

# Function to extract text
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

if uploaded_file:
    with st.spinner("Reading your PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.success("PDF uploaded and text extracted!")
    
    # Allow length option
    option = st.selectbox("🔧 Choose summary length:", ["Short", "Medium", "Long"])
    max_len = {"Short": 80, "Medium": 150, "Long": 250}[option]

    if st.button("✨ Generate Summary"):
        with st.spinner("Summarizing..."):
            text = text.replace("\n", " ")
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            summary = ""
            for chunk in chunks[:3]:  # Limit to 3 chunks for speed
                summary += summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text'] + "\n\n"
            
            st.subheader("📝 Summary:")
            st.write(summary)

            st.download_button("💾 Download Summary", summary, file_name="summary.txt")

