import chromadb
from chromadb.utils import embedding_functions
import os

def chunk_text(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def ingest_data():
    print("Initializing ChromaDB for ingestion...")
    db_path = os.path.join(os.getcwd(), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    # Delete existing to start fresh
    try:
        client.delete_collection("complaint_sops")
    except:
        pass

    collection = client.create_collection(
        name="complaint_sops",
        embedding_function=embedding_fn
    )

    # Dummy SOP Data
    documents = [
        "FAST İşlemleri: FAST (Fonların Anlık ve Sürekli Transferi) sistemi ile 7/24 para transferi yapılabilir. İşlem anında gerçekleşmezse, 'Sorgulama' adımından durum kontrol edilmelidir. 20.000 TL üzeri işlemler EFT saatlerinde gerçekleşir.",
        "EFT İptali: Yanlış hesaba yapılan EFT işlemleri için şubeye yazılı talimat verilmesi gereklidir. Mobil üzerinden iptal edilemez.",
        "Kredi Kartı İtirazı (Chargeback): Müşteri harcamayı tanımazsa, harcama itiraz formu doldurulur. Süreç 45-120 gün sürebilir.",
        "Fraud Şüphesi: Karttan bilgisi dışında işlem yapıldığını belirten müşterinin kartı derhal kullanıma kapatılmalı ve güvenlik birimine bildirilmelidir. Müşteriye yeni kart basımı önerilmelidir.",
        "Mobil Şifre Bloke: 3 kez yanlış girilen şifre sonrası bloke oluşur. Müşteri, kart bilgileri ile 'Şifre Al' menüsünden blokesini kaldırabilir.",
        "Kart Aidatı İadesi: Yasal düzenlemelere göre, aktif kullanılan ve puan kazandıran kartlar için aidat yansıtılabilir. Ancak müşteri memnuniyeti adına %50 iade veya puan yüklemesi teklif edilebilir.",
        "İnternet Arızası: Genel arıza durumunda müşteriye 'Bölgenizde çalışma var, tahmini süre 4 saat' bilgisi verilir. Bireysel arızada modem resetleme adımları iletilir."
    ]

    chunked_docs = []
    ids = []
    metadatas = []
    for doc_index, doc in enumerate(documents):
        doc_name = f"sop_{doc_index}"
        for chunk_index, chunk in enumerate(chunk_text(doc)):
            chunk_id = f"{doc_name}_chunk_{chunk_index}"
            chunked_docs.append(chunk)
            ids.append(chunk_id)
            metadatas.append(
                {
                    "source": "Bank_SOP_v1",
                    "doc_name": doc_name,
                    "chunk_id": chunk_id,
                }
            )

    print(f"Adding {len(chunked_docs)} chunks...")
    collection.add(
        documents=chunked_docs,
        ids=ids,
        metadatas=metadatas,
    )
    print("Ingestion complete. ChromaDB is ready.")

if __name__ == "__main__":
    ingest_data()
