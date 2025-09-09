RAG PDF Assistant
This is a Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents. You can upload a PDF, and the application will index its content to answer your questions based on the information within the document.
The application is built using LangChain, Ollama, ChromaDB for vector storage, and Gradio for the user interface.
Features
* PDF Upload: Upload any PDF document to serve as the knowledge base.
* Document Indexing: Automatically splits, embeds, and stores the document content in a vector database.
* Question Answering: Ask questions in natural language and receive answers sourced directly from the document.
* Clear Database: Easily clear the existing knowledge base to start fresh with a new document.
* Simple UI: A clean and user-friendly interface powered by Gradio.
Prerequisites
Before you begin, ensure you have the following installed and running:
1. Python 3.8+
2. Ollama: You need to have Ollama installed and running. You can download it from ollama.com.
3. Ollama Model: Pull the required model by running the following command in your terminal:
ollama pull gemma3:12b

Setup and Installation
   1. Clone the repository:
git clone <your-repository-url>
cd <your-repository-name>

   2. Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

   3. Install the required dependencies:
pip install -r requirements.txt

Usage
      1. Ensure Ollama is running in the background.
      2. Run the application:
python app.py

      3. Open your web browser and navigate to the local URL provided in the terminal (usually http://127.0.0.1:7860).
      4. Upload your PDF file, wait for the status message to confirm indexing, and start asking questions!
Project Structure
.
├── app.py              # Main application logic and Gradio UI
├── requirements.txt    # Python dependencies
├── .gitignore          # Files to be ignored by Git
└── README.md           # Project documentation