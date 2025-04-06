from flask import Flask, render_template, request, jsonify
import os
from rag.document_processor import DocumentProcessor
from rag.retriever import get_retriever, TFIDFRetriever, DenseRetriever
from rag.generator import get_generator

app = Flask(__name__)

# Initialize RAG components
documents_dir = os.path.join(os.path.dirname(__file__), 'files')
processor = DocumentProcessor(documents_dir)

# Get configuration from environment
retriever_type = os.getenv('RETRIEVER_TYPE', 'tfidf')
embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Get retriever based on configuration
retriever_kwargs = {}
if retriever_type.lower() == 'dense':
    retriever_kwargs = {
        'model_name': embedding_model,
        'vector_store_config': {'index_type': 'flat'}
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
    
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=top_k)
    
    # Generate response
    response_text = generator.generate_response(query, retrieved_docs)
    
    # Return response
    return jsonify({
        "query": query,
        "response": response_text,
        "retrieved_documents": retrieved_docs
    })

if __name__ == "__main__":
    app.run(debug=True)
