import json

from pypdf import PdfReader

reader = PdfReader("data/source.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap  # step forward, but re-read the last 100 characters
    return chunks


chunks = chunk_text(text)
with open("data/chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

print(len(chunks), "chunks")
print(chunks[0][:300])