from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetriever:
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
