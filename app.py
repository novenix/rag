from flask import Flask, render_template, request, jsonify
import os
from rag.document_processor import DocumentProcessor
from rag.retriever import get_retriever
from rag.generator import get_generator

app = Flask(__name__)

# Initialize RAG components
documents_dir = os.path.join(os.path.dirname(__file__), 'files')
processor = DocumentProcessor(documents_dir)

# Get retriever based on configuration (default is TFIDF for backward compatibility)
retriever_type = os.getenv('RETRIEVER_TYPE', 'tfidf')
retriever = get_retriever(retriever_type)

# Get generator
generator = get_generator(provider="together")

# Initialize document processing and indexing at application startup
# instead of using the deprecated before_first_request decorator
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
