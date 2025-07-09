import gradio as gr
import os
import fitz  # PyMuPDF
import faiss
import torch
import shutil
import tempfile
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Store session data globally
session = {
    "chunks": [],
    "embed_model": None,
    "faiss_index": None,
    "sparse_vectorizer": None,
    "sparse_matrix": None
}

# ✅ TOGETHER API SETUP
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")  # Set this in Render/Replit
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # or use falcon-40b-instruct

# ✅ TEXT CHUNKING
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def split_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text(text)
    return chunks

# ✅ EMBEDDING + INDEXING
def embed_chunks(chunks):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = model.encode(chunks, show_progress_bar=False).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, index, embeddings

def build_sparse(chunks):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix

# ✅ API COMPLETION
def query_together_api(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Answer the question using the given context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result["choices"][0]["message"]["content"]

# ✅ PDF UPLOAD HANDLER
def upload_pdf(file):
    if file is None:
        return "❌ No file uploaded."

    # Clear old data
    session["chunks"] = []
    session["embed_model"] = None
    session["faiss_index"] = None
    session["sparse_vectorizer"] = None
    session["sparse_matrix"] = None

    # Save PDF temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())

    text = extract_text(file_path)
    chunks = split_chunks(text)
    session["chunks"] = chunks

    # Build retrievers
    embed_model, faiss_index, _ = embed_chunks(chunks)
    sparse_vectorizer, sparse_matrix = build_sparse(chunks)

    # Store in session
    session["embed_model"] = embed_model
    session["faiss_index"] = faiss_index
    session["sparse_vectorizer"] = sparse_vectorizer
    session["sparse_matrix"] = sparse_matrix

    shutil.rmtree(temp_dir)  # Clean temp file
    return f"✅ Uploaded and processed {file.name} with {len(chunks)} chunks."

# ✅ ASK QUESTION HANDLER
def ask_question(query):
    if not session["chunks"]:
        return "⚠️ Please upload a PDF first."

    # Hybrid Retrieval
    query_emb = session["embed_model"].encode([query])
    _, dense_idx = session["faiss_index"].search(query_emb, 5)

    sparse_query = session["sparse_vectorizer"].transform([query])
    sparse_scores = cosine_similarity(sparse_query, session["sparse_matrix"])[0]
    sparse_idx = sparse_scores.argsort()[::-1][:5]

    hybrid_indices = list(set(dense_idx[0]).union(set(sparse_idx)))
    selected_chunks = [session["chunks"][i] for i in hybrid_indices]
    context = "\n".join(selected_chunks)

    # Create prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Get response
    try:
        answer = query_together_api(prompt)
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Hybrid RAG PDF Chatbot using Together.ai")

    file_input = gr.File(label="Upload PDF", type="binary")
    upload_btn = gr.Button("Upload & Process")
    upload_status = gr.Textbox(label="Status")

    with gr.Row():
        question = gr.Textbox(label="Ask a Question")
        ask_btn = gr.Button("Submit")

    answer = gr.Textbox(label="Answer", lines=10)

    upload_btn.click(upload_pdf, inputs=file_input, outputs=upload_status)
    ask_btn.click(ask_question, inputs=question, outputs=answer)

demo.launch()
