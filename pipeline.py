"""LLM evaluation harness: Llama 3.2 (local) vs Gemini (cloud), one step at a time.

    python pipeline.py chunks        cut the PDF into paragraphs (chunks)
    python pipeline.py exam          both models answer every question (clean exam)
    python pipeline.py exam noisy    same exam, with wrong paragraphs mixed in
    python pipeline.py retrieval     does the search put the right paragraph first?
    python pipeline.py results       collect everything into data/results_real.csv
"""
import json
import os
import sys
import time

import numpy as np
import ollama
import pandas as pd
from google.genai import errors
from pypdf import PdfReader

from models import ask_gemini, ask_ollama

EMBED_MODEL = "nomic-embed-text"
TOP_K = 3
MAX_TRIES = 6
RETRY_CODES = (429, 500, 502, 503, 504)  # overloaded / rate limited / temporary

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.
Keep the answer to 1-3 sentences.

CONTEXT:
{context}

QUESTION: {question}"""


# ---------- small helpers ----------
def load(path):
    with open(path) as f:
        return json.load(f)


def save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def embed(texts):
    """Turn texts into lists of numbers of length 1, so a dot product equals 'closeness'."""
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    vectors = np.array(response["embeddings"])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ---------- step 1: cut the PDF into paragraphs ----------
def step_chunks():
    reader = PdfReader("data/source.pdf")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    size, overlap = 800, 100
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap  # step forward, but re-read the last 100 characters
    save(chunks, "data/chunks.json")
    print(len(chunks), "chunks saved")


# ---------- step 2: the exam ----------
def get_context(item, mode, chunks, retrieval):
    if mode == "clean":
        return item["context"]
    r = retrieval[item["id"]]
    # the two highest-ranked paragraphs that are NOT correct
    noise = [c for c in r["top_chunks"] if c not in r["correct_chunks"]][:2]
    noise_text = "\n\n".join(chunks[c] for c in noise)
    return noise_text + "\n\n" + item["context"]  # wrong paragraphs first, right one last


def ask_with_retry(name, fn, prompt, qid):
    """Returns (result, retries_used). result is None if every try failed."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return fn(prompt), attempt - 1
        except errors.APIError as error:
            # a daily-quota 429 will not clear by waiting, so stop retrying at once
            if error.code == 429 and "PerDay" in str(error):
                print(f"  {name}: daily quota used up, stopping this model.")
                return None, attempt - 1
            if error.code not in RETRY_CODES:
                print(f"  {name} question {qid}: error {error.code}, skipping.")
                return None, attempt - 1
            if attempt == MAX_TRIES:
                break
            wait = min(20 * attempt, 90)  # 20s, 40s, 60s, 80s, 90s
            print(f"  {name} question {qid}: busy/limited ({error.code}), "
                  f"retry {attempt}/{MAX_TRIES - 1} in {wait}s")
            time.sleep(wait)
    print(f"  {name} question {qid}: gave up after {MAX_TRIES} tries, skipping.")
    return None, MAX_TRIES - 1


def step_exam(mode):
    golden = load("data/golden.json")
    chunks = load("data/chunks.json")
    retrieval = {r["id"]: r for r in load("data/retrieval_results.json")}
    cache = "data/answers.json" if mode == "clean" else "data/answers_noisy.json"
    answers = load(cache) if os.path.exists(cache) else {}

    models = [("llama3.2 (Ollama)", ask_ollama), ("gemini (cloud)", ask_gemini)]
    for name, fn in models:
        skipped = []
        for item in golden:
            key = f"{name}|{item['id']}"
            if key in answers:
                continue  # already answered, never spend a call twice
            prompt = PROMPT.format(
                context=get_context(item, mode, chunks, retrieval),
                question=item["question"],
            )
            result, retries = ask_with_retry(name, fn, prompt, item["id"])
            if result is None:
                skipped.append(item["id"])
                continue  # skip this question, keep going with the rest
            answers[key] = {"model": name, "question_id": item["id"],
                            "retries": retries, **result}
            save(answers, cache)  # saved after every answer
            print(name, "answered question", item["id"], "in", result["total_ms"], "ms",
                  f"(retries: {retries})")
            if "gemini" in name:
                time.sleep(2)  # small gap so we don't trigger the rate limit
        if skipped:
            print(f"{name}: skipped questions {skipped}. Run this step again to fill them in.")
    print(mode, "mode:", len(answers), "answers saved in total")


# ---------- step 3: does the search find the right paragraph? ----------
def step_retrieval():
    chunks = load("data/chunks.json")
    golden = load("data/golden.json")
    # scripts/fix_golden.py joined neighbouring chunks for these questions,
    # so both chunks count as correct for them
    extra_correct = {2: [3], 3: [8]}

    # the search model expects these two labels in front of the text
    chunk_vectors = embed(["search_document: " + c for c in chunks])

    rows = []
    for item in golden:
        correct = {item["chunk_index"], *extra_correct.get(item["id"], [])}
        question_vector = embed(["search_query: " + item["question"]])[0]
        closeness = chunk_vectors @ question_vector  # one score per chunk
        ranking = [int(i) for i in np.argsort(-closeness)]  # best match first
        rank = min(ranking.index(c) for c in correct) + 1  # 1 = the right chunk came first
        rows.append({"id": item["id"], "correct_chunks": sorted(correct),
                     "rank": rank, "top_chunks": ranking[:TOP_K]})
        print(f"Q{item['id']}: correct chunk(s) {sorted(correct)} ranked #{rank} "
              f"| top {TOP_K}: {ranking[:TOP_K]}")

    hit_at_1 = sum(r["rank"] == 1 for r in rows) / len(rows)
    hit_at_k = sum(r["rank"] <= TOP_K for r in rows) / len(rows)
    mrr = sum(1 / r["rank"] for r in rows) / len(rows)
    print(f"\nCorrect chunk ranked #1: {hit_at_1:.0%}")
    print(f"Correct chunk in top {TOP_K}: {hit_at_k:.0%}")
    print(f"MRR (average of 1/rank, 1.0 is perfect): {mrr:.2f}")
    save(rows, "data/retrieval_results.json")


# ---------- step 4: collect everything into one table ----------
def step_results():
    golden = {item["id"]: item for item in load("data/golden.json")}
    clean = load("data/answers.json")
    noisy = load("data/answers_noisy.json")
    labels = load("data/human_labels.json")
    retrieval = {r["id"]: r for r in load("data/retrieval_results.json")}

    def closeness(text_a, text_b):
        a, b = embed([text_a, text_b])
        return float(a @ b)  # 1.0 means the same meaning

    rows = []
    for key, ans in clean.items():
        qid = ans["question_id"]
        expected = golden[qid]["expected_answer"]
        similarity = closeness(ans["answer"], expected)
        noise_drop = None
        if key in noisy:  # only if this answer also exists in the noisy exam
            noise_drop = similarity - closeness(noisy[key]["answer"], expected)
        rows.append({
            "model": ans["model"],
            "question_id": qid,
            "answer_similarity": round(similarity, 3),
            "faithful": int(labels[key]),  # your own label: 1 = nothing made up
            "noise_drop": None if noise_drop is None else round(noise_drop, 3),
            "retrieval_rank": retrieval[qid]["rank"],
            "latency_ms": ans["total_ms"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/results_real.csv", index=False)
    print(df.groupby("model").agg(
        answer_similarity=("answer_similarity", "mean"),
        faithful=("faithful", "sum"),
        noise_drop=("noise_drop", "mean"),
        median_latency_ms=("latency_ms", "median"),
    ).round(3))
    print("\nrows:", len(df), "| answers with a noise result:", df["noise_drop"].notna().sum())


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    mode = sys.argv[2] if len(sys.argv) > 2 else "clean"
    if step == "chunks":
        step_chunks()
    elif step == "exam" and mode in ("clean", "noisy"):
        step_exam(mode)
    elif step == "retrieval":
        step_retrieval()
    elif step == "results":
        step_results()
    else:
        print(__doc__)