import json

import ollama
from pydantic import BaseModel

chunks = json.load(open("data/chunks.json"))
PICK = [1, 4, 7, 10, 13, 16, 19, 22, 24, 26]  # ten chunks spread across the document


class QA(BaseModel):
    question: str
    expected_answer: str


PROMPT = """Using ONLY the text below, write one question a student might ask,
and a short correct answer (1-2 sentences) that is stated in the text.

TEXT:
{chunk}"""

golden = []
for n, idx in enumerate(PICK, start=1):
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": PROMPT.format(chunk=chunks[idx])}],
        format=QA.model_json_schema(),  # forces valid JSON with our two keys
    )
    qa = QA.model_validate_json(response["message"]["content"])
    golden.append({
        "id": n,
        "chunk_index": idx,
        "question": qa.question,
        "expected_answer": qa.expected_answer,
        "context": chunks[idx],
    })
    print(n, qa.question)
    with open("data/golden_draft.json", "w") as f:  # saved after every item
        json.dump(golden, f, indent=2)