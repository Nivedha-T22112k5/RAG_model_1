import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

app = Flask(__name__)

# Directory setup
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
PERSIST_DIRECTORY = os.path.join(os.getcwd(), 'persist')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit

# LangChain automatically detects GEMINI_API_KEY from environment
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

def process_and_index_file(file_path):
    """Parses, splits, and indexes uploaded file into ChromaDB."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or TXT files.")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=splits)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_path)

        try:
            process_and_index_file(saved_path)
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return jsonify({'message': f'File "{filename}" processed and added to ChromaDB!'})
        except Exception as e:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_rag():
    data = request.get_json() or {}
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({'error': 'Query text is required'}), 400

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context pieces to answer the question. "
            "If you don't know the answer, say that you don't know.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": user_query})
        
        return jsonify({
            'answer': response['answer'],
            'sources': [doc.page_content for doc in response.get('context', [])]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)