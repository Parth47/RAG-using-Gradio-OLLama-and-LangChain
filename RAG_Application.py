# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import gradio as gr
import torch
import shutil
import os
import warnings

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Suppress unnecessary warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')


# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
PERSIST_DIRECTORY = "./chroma_db_temp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Global variable for the QA chain
qa_chain = None


# ==============================================================================
# 3. MODEL INITIALIZATION
# ==============================================================================
def initialize_llm():
    """Initializes and returns the Ollama LLM."""
    try:
        llm = Ollama(
            model="gemma3:12b",
            temperature=0.1,
            system="You are a helpful assistant. Answer the user's questions based on the provided context. If you don't find the answer from the context, mention explicitly that you do not know the answer."
        )
        # A small check to see if the model is available
        llm.invoke("Hi")
        print("Ollama LLM initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        return None

def get_embedding_model():
    """Initializes and returns the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': DEVICE}
    )

# Initialize models globally
llm = initialize_llm()
embedding_model = get_embedding_model()


# ==============================================================================
# 4. CORE RAG PIPELINE
# ==============================================================================
def process_and_embed_pdf(file_path):
    """
    Loads a PDF, splits it, creates embeddings, and saves them to a vector store.
    Returns a status message (string) indicating success or failure.
    """
    global qa_chain

    try:
        # 1. Clean up any existing database
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            print("Removed old vector database.")

        # 2. Load the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages.")

        # 3. Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split document into {len(chunks)} chunks.")

        # 4. Create and persist the vector database
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # 5. Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )
        return f"Successfully indexed '{os.path.basename(file_path)}'. You can now ask questions."

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        # Clean up if the process failed midway
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        qa_chain = None
        return f"An error occurred while processing the PDF: {e}"


# ==============================================================================
# 5. GRADIO UI HANDLER FUNCTIONS
# ==============================================================================
def handle_pdf_upload(file):
    """Gradio handler for PDF upload. Updates UI based on processing status."""
    if llm is None:
        gr.Warning("Ollama is not running! Please start it and restart the app.")
        return "Ollama not found.", gr.update(), gr.update(), gr.update()

    if file is not None:
        status_message = process_and_embed_pdf(file.name)
        
        if "Successfully indexed" in status_message:
            return (
                status_message,
                gr.update(visible=True, value=[]),  # Show and clear chatbot
                gr.update(visible=True),           # Show query row
                gr.update(interactive=True)        # Enable delete button
            )
        else:
            # An error occurred
            return (
                status_message,
                gr.update(visible=False),          # Keep chatbot hidden
                gr.update(visible=False),          # Keep query row hidden
                gr.update(interactive=False)       # Keep delete button disabled
            )
    
    return "Please upload a PDF.", gr.update(), gr.update(), gr.update()

def handle_user_query(query, history):
    """Gradio handler for the chatbot query."""
    if qa_chain:
        try:
            response = qa_chain.invoke(query)
            history.append((query, response['result']))
        except Exception as e:
            history.append((query, f"An error occurred: {e}"))
    else:
        history.append((query, "Error: Please upload and process a PDF first."))
    return "", history  # Return an empty string to clear the query box

def handle_delete_database():
    """Gradio handler for deleting the vector database and resetting the UI."""
    global qa_chain
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            qa_chain = None
            return (
                "Database deleted. You can now upload a new PDF.",
                gr.update(value=[], visible=False),  # Clear and hide chatbot
                gr.update(visible=False),           # Hide query row
                gr.update(interactive=False),       # Disable delete button
                gr.update(value=None)               # Clear the file upload component
            )
        except Exception as e:
            return f"Error deleting database: {e}", gr.update(), gr.update(), gr.update(), gr.update()
            
    return "No database found to delete.", gr.update(), gr.update(), gr.update(), gr.update()


# ==============================================================================
# 6. GRADIO UI DEFINITION
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="RAG PDF Assistant") as demo:
    gr.Markdown("# RAG Assistant for Your PDFs 📄")
    gr.Markdown("1. Upload a PDF. 2. Wait for indexing to complete. 3. Ask questions about the document.")

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        delete_btn = gr.Button("🗑️ Clear Database", variant="stop", interactive=False)

    status_box = gr.Textbox("Please upload your PDF file to start.", label="Status", interactive=False)
    
    # Chatbot interface (initially hidden)
    chatbot = gr.Chatbot(label="Chat History", visible=False, height=400)
    
    with gr.Row(visible=False) as query_row:
        query_box = gr.Textbox(label="Enter your query", scale=4, placeholder="Ask something about the document...")
        submit_btn = gr.Button("Ask", variant="primary", scale=1)
    
    # Event Handlers
    pdf_upload.upload(
        handle_pdf_upload,
        inputs=[pdf_upload],
        outputs=[status_box, chatbot, query_row, delete_btn]
    )
    
    submit_btn.click(
        handle_user_query,
        inputs=[query_box, chatbot],
        outputs=[query_box, chatbot]
    )

    query_box.submit(
        handle_user_query,
        inputs=[query_box, chatbot],
        outputs=[query_box, chatbot]
    )
    
    delete_btn.click(
        handle_delete_database,
        outputs=[status_box, chatbot, query_row, delete_btn, pdf_upload]
    )


# ==============================================================================
# 7. APPLICATION LAUNCH
# ==============================================================================
if __name__ == "__main__":
    if llm is None:
        print("Could not start the Gradio app because Ollama is not available.")
        print("Please ensure Ollama is installed, running, and the 'gemma3:12b' model is downloaded.")
    else:
        print("Launching Gradio Interface...")
        demo.launch(share=False)
