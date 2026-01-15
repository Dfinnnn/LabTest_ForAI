import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# ---------------------------------------------------------
# Step 1: NLTK Setup & Page Config
# ---------------------------------------------------------
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
    # Step 3: Tokenization & User Selection
    # ---------------------------------------------------------
    st.subheader("Step 3: Sentence Selection")
    
    # Split text into sentences (chunks)
    sentences = sent_tokenize(raw_text)
    total_sentences = len(sentences)
    
    st.metric("Total Sentences Found", total_sentences)

    if total_sentences > 0:
        # --- NEW CODE: User Inputs for Indices ---
        st.write("### Choose the range of sentences to display:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Default start is 58 (as per lab req), but clamped to valid range
            default_start = 58 if total_sentences > 58 else 0
            start_idx = st.number_input(
                "Start Index", 
                min_value=0, 
                max_value=total_sentences - 1, 
                value=default_start,
                step=1
            )
            
        with col2:
            # Default end is 68, but clamped to valid range
            default_end = 68 if total_sentences > 68 else total_sentences - 1
            # Ensure max_value doesn't crash if start_idx changed
            end_idx = st.number_input(
                "End Index", 
                min_value=start_idx, 
                max_value=total_sentences - 1, 
                value=max(start_idx, default_end),
                step=1
            )
        
        # Display the specific range
        st.divider()
        st.info(f"Displaying sentences from index {start_idx} to {end_idx}:")
        
        for i in range(start_idx, end_idx + 1):
            st.markdown(f"**Index {i}:** {sentences[i]}")
            
    else:
        st.warning("No text could be extracted or tokenized from this PDF.")

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