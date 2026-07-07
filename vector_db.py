import os
import csv

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_PATH, COLLECTION_NAME


class VectorDB:
    def __init__(self, persist_path=CHROMA_PERSIST_PATH, chunks=None):
        self.client = chromadb.PersistentClient(path=persist_path)
        existing_collections = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing_collections:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            model_name = self.collection.metadata["embedding_model"]
            self.model = SentenceTransformer(model_name)
            print(f"Base rechargée. Modèle d'embedding utilisé : {model_name}")

        elif chunks is not None:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"embedding_model": EMBEDDING_MODEL_NAME},
            )
            self._index(chunks)
            print(f"Base créée et indexée avec {len(chunks)} chunks.")

        else:
            raise ValueError(
                "Aucune base existante à ce chemin et aucun chunk fourni. "
                "Impossible de démarrer."
            )

    def _index(self, chunks):
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"]} for c in chunks]

        embeddings = self.model.encode(
            documents,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def retrieve(self, question, n=3):
        question_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=n,
        )
        return results


def load_chunks_from_csv(path):
    chunks = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "source": row["source"],
                }
            )
    return chunks