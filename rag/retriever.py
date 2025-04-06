from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer

class Retriever(ABC):
    """Abstract base class for document retrieval."""
    
    @abstractmethod
    def index_documents(self, document_chunks: List[Dict]):
        """Index the document chunks for retrieval."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top-k relevant document chunks for a query."""
        pass

class TFIDFRetriever(Retriever):
    def __init__(self):
        """Initialize the TF-IDF retriever."""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_chunks = []
        self.tfidf_matrix = None
        
    def index_documents(self, document_chunks: List[Dict]):
        """
        Index the document chunks using TF-IDF.
        
        Args:
            document_chunks: List of document chunks with text and metadata
        """
        self.document_chunks = document_chunks
        texts = [chunk["text"] for chunk in document_chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k relevant document chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of most relevant document chunks with scores
        """
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                "text": self.document_chunks[idx]["text"],
                "metadata": self.document_chunks[idx]["metadata"],
                "score": float(similarities[idx])
            })
        
        return results

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, document_chunks: List[Dict]):
        """Add vectors and their associated documents to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for the top-k most similar vectors."""
        pass

class FAISSVectorStore(VectorStore):
    """Vector store implementation using FAISS."""
    
    def __init__(self, dimension: int):
        """Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of the vectors to store
        """
        # Create a flat (brute-force) L2 index
        self.index = faiss.IndexFlatL2(dimension)
        self.document_chunks = []
        
    def add_vectors(self, vectors: np.ndarray, document_chunks: List[Dict]):
        """Add vectors and their associated documents to the store."""
        # Make sure vectors are in float32 format
        vectors = vectors.astype(np.float32)
        
        # Add vectors to the index
        self.index.add(vectors)
        self.document_chunks = document_chunks
        
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for the top-k most similar vectors."""
        # Make sure query vector is in float32 format and reshaped for FAISS
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # Perform the search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.document_chunks):  # Ensure index is valid
                results.append({
                    "text": self.document_chunks[idx]["text"],
                    "metadata": self.document_chunks[idx]["metadata"],
                    "score": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                })
        
        return results

class DenseRetriever(Retriever):
    """Retriever that uses dense embeddings and a vector store."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_store: VectorStore = None):
        """Initialize the dense retriever with embedding model and vector store.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            vector_store: Vector store to use (if None, will create a FAISS store)
        """
        self.model = SentenceTransformer(model_name)
        self.document_chunks = []
        
        # If no vector store is provided, create a FAISS store with the model's output dimension
        if vector_store is None:
            self.vector_store = FAISSVectorStore(self.model.get_sentence_embedding_dimension())
        else:
            self.vector_store = vector_store
        
    def index_documents(self, document_chunks: List[Dict]):
        """Index the document chunks using embeddings and the vector store."""
        self.document_chunks = document_chunks
        texts = [chunk["text"] for chunk in document_chunks]
        
        # Generate embeddings for all document chunks
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, document_chunks)
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top-k relevant document chunks for a query."""
        # Generate embedding for the query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search the vector store
        return self.vector_store.search(query_embedding, top_k)

def get_retriever(retriever_type: str = "tfidf", **kwargs) -> Retriever:
    """Factory function to create a retriever.
    
    Args:
        retriever_type: Type of retriever to create ('tfidf' or 'dense')
        **kwargs: Additional arguments to pass to the retriever constructor
    
    Returns:
        An instance of a Retriever subclass
    """
    if retriever_type.lower() == "tfidf":
        return TFIDFRetriever()
    elif retriever_type.lower() == "dense":
        return DenseRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
