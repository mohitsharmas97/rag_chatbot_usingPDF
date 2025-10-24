import streamlit as st
from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

# Configure Google AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# --- Cached Resources ---

@st.cache_resource
def load_embeddings_model():
    """Loads the sentence transformer model once."""
    with st.spinner("Loading embeddings model..."):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

@st.cache_resource
def get_gemini_model():
    """Initializes Gemini model once."""
    return genai.GenerativeModel('gemini-2.0-flash-exp')

# --- Core Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    total_pages = 0
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                total_pages += 1
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
            continue
    
    if total_pages > 0:
        st.info(f"📄 Extracted text from {total_pages} pages across {len(pdf_docs)} PDF(s)")
    
    return text

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks."""
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at a sentence boundary
        if end < text_length:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            boundary = max(last_period, last_newline, last_space)
            if boundary > chunk_size * 0.5:  # Only adjust if we're not cutting too much
                end = start + boundary + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def create_vector_store(text_chunks):
    """Creates FAISS index from text chunks."""
    if not text_chunks:
        st.warning("No text chunks to process.")
        return None, None
    
    model = load_embeddings_model()
    
    progress_bar = st.progress(0, text="Creating embeddings...")
    
    try:
        # Generate embeddings
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            
            progress = min((i + batch_size) / len(text_chunks), 1.0)
            progress_bar.progress(
                progress,
                text=f"Processing {min(i + batch_size, len(text_chunks))}/{len(text_chunks)} chunks..."
            )
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        progress_bar.success(f"✅ Created vector store with {len(text_chunks)} chunks!")
        
        # Save to disk
        faiss.write_index(index, "faiss_index.bin")
        with open("text_chunks.pkl", "wb") as f:
            pickle.dump(text_chunks, f)
        
        return index, text_chunks
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None

def search_similar_chunks(query, index, text_chunks, k=4):
    """Searches for similar chunks using FAISS."""
    model = load_embeddings_model()
    
    # Encode query
    query_embedding = model.encode([query]).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    # Get relevant chunks
    results = []
    for idx in indices[0]:
        if idx < len(text_chunks):
            results.append(text_chunks[idx])
    
    return results

def get_answer_from_gemini(question, context_chunks):
    """Gets answer from Gemini using context."""
    model = get_gemini_model()
    
    # Prepare context
    context = "\n\n---\n\n".join(context_chunks)
    
    # Create prompt
    prompt = f"""You are a helpful assistant answering questions based on provided document context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- If the answer is not in the context, say "I cannot find this information in the provided documents"
- Be detailed and specific in your answer
- Quote relevant parts if helpful

Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

def get_answer(user_question, index, text_chunks):
    """Main function to get answer for user question."""
    if index is None or text_chunks is None:
        st.error("Please upload and process PDFs first.")
        return
    
    try:
        # Search for relevant chunks
        with st.spinner("Searching documents..."):
            relevant_chunks = search_similar_chunks(user_question, index, text_chunks, k=4)
        
        if not relevant_chunks:
            st.write("*Reply:* I couldn't find relevant information in the documents.")
            return
        
        # Get answer from Gemini
        with st.spinner("Generating answer..."):
            answer = get_answer_from_gemini(user_question, relevant_chunks)
        
        # Display answer
        st.markdown("### 💬 Answer:")
        st.write(answer)
        
        # Show source chunks
        with st.expander("📚 View Source Chunks"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"*Chunk {i}:*")
                st.text_area(
                    f"chunk_{i}",
                    chunk[:500] + ("..." if len(chunk) > 500 else ""),
                    height=150,
                    label_visibility="collapsed"
                )
                if i < len(relevant_chunks):
                    st.divider()
                
    except Exception as e:
        st.error(f"Error generating answer: {e}")

def load_existing_index():
    """Loads existing FAISS index and chunks from disk."""
    try:
        if os.path.exists("faiss_index.bin") and os.path.exists("text_chunks.pkl"):
            index = faiss.read_index("faiss_index.bin")
            with open("text_chunks.pkl", "rb") as f:
                text_chunks = pickle.load(f)
            return index, text_chunks
    except Exception as e:
        st.warning(f"Could not load existing index: {e}")
    return None, None

# --- Main App ---

def main():
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Chat with PDF using Gemini")
    st.markdown("Upload PDF files and ask questions about their content using Google's Gemini AI.")
    
    # Initialize session state
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Try to load existing index on startup
    if st.session_state.index is None:
        index, chunks = load_existing_index()
        if index is not None:
            st.session_state.index = index
            st.session_state.text_chunks = chunks
            st.session_state.processed_files = ["Previously processed documents"]
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Process", type="primary", use_container_width=True):
                if not pdf_docs:
                    st.warning("Please upload at least one PDF file.")
                else:
                    with st.spinner("Processing documents..."):
                        # Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text or len(raw_text.strip()) == 0:
                            st.error("No text could be extracted from the PDFs.")
                        else:
                            # Create chunks
                            text_chunks = split_text_into_chunks(raw_text)
                            st.info(f"📝 Created {len(text_chunks)} text chunks")
                            
                            # Create vector store
                            index, chunks = create_vector_store(text_chunks)
                            
                            if index is not None:
                                st.session_state.index = index
                                st.session_state.text_chunks = chunks
                                st.session_state.processed_files = [pdf.name for pdf in pdf_docs]
                                st.success("✅ Documents processed successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to create vector store.")
        
        with col2:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.index = None
                st.session_state.text_chunks = None
                st.session_state.processed_files = []
                
                # Clean up files
                if os.path.exists("faiss_index.bin"):
                    os.remove("faiss_index.bin")
                if os.path.exists("text_chunks.pkl"):
                    os.remove("text_chunks.pkl")
                
                st.success("Cleared all data!")
                st.rerun()
        
        # Show processed files
        if st.session_state.processed_files:
            st.divider()
            st.subheader("✅ Processed Files:")
            for filename in st.session_state.processed_files:
                st.text(f"• {filename}")
            
            if st.session_state.text_chunks:
                st.metric("Total Chunks", len(st.session_state.text_chunks))
    
    # Main chat interface
    st.divider()
    
    if st.session_state.index is None:
        st.info("👈 Upload PDF files using the sidebar to get started.")
        
        st.markdown("""
        ### How to use:
        1. Upload one or more PDF files using the sidebar
        2. Click *Process* to analyze the documents
        3. Ask questions about the content
        4. Get AI-powered answers based on your documents
        """)
    else:
        user_question = st.text_input(
            "💬 Ask a question about your documents:",
            placeholder="e.g., What is the main topic discussed in the document?"
        )
        
        if user_question:
            get_answer(user_question, st.session_state.index, st.session_state.text_chunks)
        
        # Add some example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - What is the main topic of this document?
            - Can you summarize the key points?
            - What are the conclusions mentioned?
            - Are there any specific dates or numbers mentioned?
            - Who are the main people/entities discussed?
            """)

if __name__ == "__main__":
    main()