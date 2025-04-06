import os
import json
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Generator(ABC):
    """Abstract base class for text generators."""
    
    @abstractmethod
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate a response based on query and context documents."""
        pass
    
    def format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Combine context documents into a single string."""
        return "\n\n".join([doc["text"] for doc in context_docs])
    
    def create_base_prompt(self, query: str, context: str) -> str:
        """Create a base prompt template."""
        return f"""
        Answer the question based on the context provided. If you cannot answer based on the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """


class OpenAIGenerator(Generator):
    """Generator that uses OpenAI's API."""
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.2, max_tokens=500):
        """Initialize the OpenAI generator."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = self.api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using OpenAI based on the query and retrieved documents.
        """
        context = self.format_context(context_docs)
        prompt = self.create_base_prompt(query, context)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information about HistoriaCard, a Mexican fintech company, based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response with OpenAI: {str(e)}"


class TogetherAIGenerator(Generator):
    """Generator that uses Together.ai's API."""
    
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=0.2, max_tokens=500):
        """Initialize the Together.ai generator."""
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using Together.ai based on the query and retrieved documents.
        """
        context = self.format_context(context_docs)
        prompt = self.create_base_prompt(query, context)
        
        system_message = "You are a helpful assistant that provides accurate information about HistoriaCard, a Mexican fintech company, based only on the provided context."
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise exception for HTTP errors
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error generating response with Together.ai: {str(e)}"


# Factory function to get the appropriate generator
def get_generator(provider="openai", **kwargs):
    """
    Factory function to create and return the specified generator.
    
    Args:
        provider: The name of the provider to use ('openai', 'together', etc.)
        **kwargs: Additional arguments to pass to the generator constructor
    
    Returns:
        An instance of a Generator subclass
    """
    if provider.lower() == "openai":
        return OpenAIGenerator(**kwargs)
    elif provider.lower() == "together":
        return TogetherAIGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
