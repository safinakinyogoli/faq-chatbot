# rag_engine.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from huggingface_hub import InferenceClient

class RAGEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.texts = []
        self.client = None
        self.initialize_rag()
    
    def initialize_rag(self):
        """Initialize the RAG system"""
        try:
            print("Initializing RAG Engine...")
            
            # Check API key
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("HUGGINGFACE_API_KEY not set")
            
            # Load embedding model
            print("Loading embedding model...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load FAISS index
            print("Loading FAISS index...")
            faiss_path = "faiss_index_nhif"
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
            
            self.index = faiss.read_index(os.path.join(faiss_path, "index.faiss"))
            
            # Load texts
            with open(os.path.join(faiss_path, "texts.pkl"), "rb") as f:
                self.texts = pickle.load(f)
            
            # Setup HuggingFace client
            print("Setting up HuggingFace client...")
            self.client = InferenceClient(
                model="mistralai/Mistral-7B-Instruct-v0.1",
                token=api_key
            )
            
            print("✅ RAG Engine ready!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def get_response(self, question):
        """Get response for user question"""
        try:
            # Encode question
            question_embedding = self.model.encode([question])
            
            # Search similar documents
            distances, indices = self.index.search(
                question_embedding.astype('float32'), 
                k=3
            )
            
            # Get relevant texts
            relevant_texts = []
            for i in indices[0]:
                if i < len(self.texts):
                    relevant_texts.append(self.texts[i])
            
            context = "\n\n".join(relevant_texts)
            
            # Create prompt
            prompt = f"""<s>[INST] You are a helpful healthcare assistant for NHIF. 

Context: {context}

Question: {question}

Answer concisely: [/INST]"""
            
            # Generate response
            response = self.client.text_generation(
                prompt,
                max_new_tokens=500,
                temperature=0.7
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"