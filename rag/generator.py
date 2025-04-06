import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables (for OpenAI API key)
load_dotenv()

class OpenAIGenerator:
    def __init__(self):
        """Initialize the OpenAI generator."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = self.api_key
        
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using OpenAI based on the query and retrieved documents.
        
        Args:
            query: User query
            context_docs: Retrieved relevant documents
            
        Returns:
            Generated response text
        """
        # Combine context documents into a single string
        context = "\n\n".join([doc["text"] for doc in context_docs])
        
        # Create prompt for OpenAI
        prompt = f"""
        Answer the question based on the context provided. If you cannot answer based on the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        # Call OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information about HistoriaCard, a Mexican fintech company, based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
