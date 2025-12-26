import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Tuple, Dict, Any

class RAGManager:
    def __init__(self):
        # Initialize ChromaDB Client
        # Persistent storage in ./chroma_db
        db_path = os.path.join(os.getcwd(), "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use simple default embedding function (all-MiniLM-L6-v2)
        # Note: In production for Turkish, a multilingual model like 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' is better
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction() 
        
        self.collection = self.client.get_or_create_collection(
            name="complaint_sops",
            embedding_function=self.embedding_fn
        )

    def retrieve(self, query: str, n_results: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            # Flatten results list
            if results["documents"]:
                documents = results["documents"][0]
                metadatas = results.get("metadatas", [[]])[0]
                return documents, metadatas
            return [], []
        except Exception as e:
            print(f"RAG Retrieve Error: {e}")
            return [], []

rag_manager = RAGManager()
