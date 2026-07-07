import json
from groq import Groq

MODERATOR_MODEL_NAME = "openai/gpt-oss-safeguard-20b"
SYSTEM_PROMPT_PATH = "prompts/system_moderator.txt"


class Moderator:
    def __init__(self, client: Groq):
        self.client = client
        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question: str) -> dict:
        response = self.client.chat.completions.create(
            model=MODERATOR_MODEL_NAME,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
        )
        raw = response.choices[0].message.content
        return json.loads(raw)