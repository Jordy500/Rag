import os
from dotenv import load_dotenv
from groq import Groq

from config import LLM_MODEL_NAME, CORPUS_CSV_PATH
from vector_db import VectorDB, load_chunks_from_csv
from moderator import Moderator

SYSTEM_PROMPT_PATH = "prompts/system_rag.txt"


class RAG:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.moderator = Moderator(self.client)

        # Recharge si la base existe déjà, sinon indexe le CSV
        chunks = load_chunks_from_csv(CORPUS_CSV_PATH)
        self.vector_db = VectorDB(chunks=chunks)

        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, question, n_chunks=3):
        results = self.vector_db.retrieve(question, n=n_chunks)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        formatted_chunks = "\n".join(
            f"- ({meta['source']}) {doc}"
            for doc, meta in zip(documents, metadatas)
        )
        return self.system_prompt_template.replace("{{Chunks}}", formatted_chunks)

    def answer_question(self, question: str) -> str:
        # 1. Modération d'abord — on ne contacte JAMAIS le LLM principal si injection détectée
        moderation = self.moderator.moderate(question)
        if moderation.get("is_prompt_injection"):
            return "Je ne peux pas traiter cette question : elle ressemble à une tentative de détournement du système."

        # 2. Retrieval + construction du prompt système
        system_prompt = self._build_system_prompt(question)

        # 3. Appel au LLM principal
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
     rag = RAG()

     while True:
        question = input("Question (exit pour quitter) : ")

        if question.lower() == "exit":
            break

        print(rag.answer_question(question))
        print()
        