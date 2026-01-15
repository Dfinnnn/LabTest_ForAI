import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# ---------------------------------------------------------
# Step 1: NLTK Setup & Page Config
# ---------------------------------------------------------
# Download necessary NLTK data quietly
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="Text Chunking Web App", layout="wide")
st.title("Question 4: Text Chunking with NLTK")
st.markdown("### Semantic Text Chunking from PDF")

# ---------------------------------------------------------
# Step 2: PDF Import & Extraction Function
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """Reads a PDF file and returns the full text content."""
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

# Upload Widget
uploaded_file = st.file_uploader("Step 1: Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text using the helper function
    with st.spinner("Step 2: Extracting text..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    
    st.success("PDF Extracted Successfully!")
    
    # ---------------------------------------------------------
    # Step 3: Preprocessing & Display Sample (Indices 58-68)
    # ---------------------------------------------------------
    st.subheader("Step 3: Sample Extracted Text (Indices 58-68)")
    
    # Split text into sentences (chunks)
    sentences = sent_tokenize(raw_text)
    total_sentences = len(sentences)
    
    st.metric("Total Sentences Found", total_sentences)

    # Check if we have enough sentences to meet the specific question requirement
    start_idx = 58
    end_idx = 68
    
    if total_sentences > start_idx:
        # Create a container for the requested indices
        st.info(f"Displaying sentences from index {start_idx} to {end_idx}:")
        
        # Display the specific range 58 to 68
        for i in range(start_idx, min(end_idx + 1, total_sentences)):
            st.markdown(f"**Index {i}:** {sentences[i]}")
    else:
        st.warning(f"The PDF is too short! It only has {total_sentences} sentences. Cannot show indices 58-68.")
        # Fallback: Show first 10 sentences if file is short
        st.write("Displaying first 5 sentences instead:")
        for i, sent in enumerate(sentences[:5]):
            st.write(f"**Index {i}:** {sent}")

    # ---------------------------------------------------------
    # Step 4: Semantic Chunking Analysis (Full View)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Step 4: Full Semantic Chunking Analysis")
    st.write("This section tokenizes the entire text based on NLTK's semantic sentence splitter.")
    
    with st.expander("Click to view all chunks"):
        for i, sentence in enumerate(sentences):
            st.text(f"Chunk {i}: {sentence}")

else:
    st.info("Waiting for PDF upload...")