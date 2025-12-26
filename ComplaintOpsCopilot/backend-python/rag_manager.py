import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict

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

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, str]]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            # Flatten results list
            if results["documents"]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                return [
                    {
                        "snippet": doc,
                        "source": metadata.get("source", "unknown"),
                        "doc_name": metadata.get("doc_name", "unknown"),
                        "chunk_id": metadata.get("chunk_id", "unknown"),
                    }
                    for doc, metadata in zip(documents, metadatas)
                ]
            return []
        except Exception as e:
            print(f"RAG Retrieve Error: {e}")
            return [], []

rag_manager = RAGManager()
