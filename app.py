from flask import Flask, render_template, request, jsonify, session
import os
from rag.document_processor import DocumentProcessor
from rag.retriever import get_retriever, TFIDFRetriever, DenseRetriever
from rag.generator import get_generator
from rag.dialog_state import DialogStateManager

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key')  # Set a secret key for sessions

# Initialize RAG components
documents_dir = os.path.join(os.path.dirname(__file__), 'files')
processor = DocumentProcessor(documents_dir)

# Initialize the dialog state manager
dialog_manager = DialogStateManager()

# Get configuration from environment
retriever_type = os.getenv('RETRIEVER_TYPE', 'tfidf')
embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
use_reranking = os.getenv('USE_RERANKING', 'false').lower() == 'true'

# Get retriever based on configuration
retriever_kwargs = {}
print(f"Using use_reranking type: {use_reranking}")
if retriever_type.lower() == 'dense':
    retriever_kwargs = {
        'model_name': embedding_model,
        'vector_store_config': {'index_type': 'flat'}
    }
    # show in terminal the retriever type
    print(f"Using dense retriever with embedding model: {embedding_model}")
    if use_reranking:
        retriever_type = 'rerank'
        retriever_kwargs = {
            'base_retriever_type': 'dense',
            'base_retriever_kwargs': retriever_kwargs,
            'cross_encoder_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'initial_top_k': 10
        }
        
elif retriever_type.lower() == 'hybrid':
    # Create individual retrievers
    tfidf_retriever = TFIDFRetriever()
    dense_retriever = DenseRetriever(
        model_name=embedding_model
    )
    
    # Configure hybrid with both retrievers
    retriever_kwargs = {
        'retrievers': {
            'tfidf': tfidf_retriever,
            'dense': dense_retriever
        },
        'weights': {'tfidf': 0.3, 'dense': 0.7}  # Giving more weight to dense retrieval
    }
    
    if use_reranking:
        retriever_type = 'rerank'
        retriever_kwargs = {
            'base_retriever_type': 'hybrid',
            'base_retriever_kwargs': retriever_kwargs,
            'cross_encoder_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'initial_top_k': 10
        }

retriever = get_retriever(retriever_type, **retriever_kwargs)

# Get generator
generator = get_generator(provider="together")

# Initialize document processing and indexing at application startup
documents = processor.load_documents()
chunks = processor.chunk_documents()
retriever.index_documents(chunks)
print(f"Indexed {len(chunks)} document chunks from {len(documents)} documents using {retriever_type} retriever")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/api/rag/query", methods=["POST"])
def query_rag():
    """API endpoint for querying the RAG system."""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"}), 400
    
    query = data['query']
    top_k = data.get('top_k', 3)
    session_id = data.get('session_id')
    
    # Create a new session if one doesn't exist
    if not session_id:
        session_id = dialog_manager.create_session()
    
    # Get conversation history
    conversation_history = dialog_manager.format_history_for_llm(session_id, include_last_n_turns=3)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=top_k)
    
    # Generate response
    response_text = generator.generate_response(query, retrieved_docs, conversation_history)
    
    # Add to conversation history
    dialog_manager.add_to_history(session_id, "user", query)
    dialog_manager.add_to_history(session_id, "assistant", response_text)
    
    # Return response with session ID
    return jsonify({
        "query": query,
        "response": response_text,
        "retrieved_documents": retrieved_docs,
        "session_id": session_id
    })

# Add an endpoint to clear conversation history
@app.route("/api/conversation/clear", methods=["POST"])
def clear_conversation():
    """API endpoint to clear conversation history."""
    data = request.json
    if not data or 'session_id' not in data:
        return jsonify({"error": "Missing session_id parameter"}), 400
    
    session_id = data['session_id']
    dialog_manager.clear_history(session_id)
    
    return jsonify({"status": "success", "message": "Conversation history cleared"})

if __name__ == "__main__":
    app.run(debug=True)
