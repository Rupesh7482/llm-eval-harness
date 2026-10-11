import json

import numpy as np
import ollama
import pandas as pd

golden = {item["id"]: item for item in json.load(open("data/golden.json"))}
clean = json.load(open("data/answers.json"))
noisy = json.load(open("data/answers_noisy.json"))
labels = json.load(open("data/human_labels.json"))
retrieval = {r["id"]: r for r in json.load(open("data/retrieval_results.json"))}


def closeness(text_a, text_b):
    response = ollama.embed(model="nomic-embed-text", input=[text_a, text_b])
    a, b = np.array(response["embeddings"])
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # 1.0 = same meaning


rows = []
for key, ans in clean.items():
    qid = ans["question_id"]
    expected = golden[qid]["expected_answer"]
    similarity = closeness(ans["answer"], expected)
    noise_drop = None
    if key in noisy:  # only if Gemini/Llama already answered the noisy exam
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
print(df.groupby("model")[["answer_similarity", "faithful", "noise_drop", "latency_ms"]].mean().round(3))
print("\nrows:", len(df), "| answers with a noise result:", df["noise_drop"].notna().sum())