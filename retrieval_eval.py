import json

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
K = 3  # "top 3" results

chunks = json.load(open("data/chunks.json"))
golden = json.load(open("data/golden.json"))

# fix_golden.py joined neighbouring chunks for these questions, so both chunks count as correct
EXTRA_CORRECT = {2: [3], 3: [8]}


def embed(texts):
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    vectors = np.array(response["embeddings"])
    # make every vector length 1, so a dot product equals cosine similarity (closeness)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# this model was trained to expect these two labels in front of the text
chunk_vectors = embed(["search_document: " + c for c in chunks])

rows = []
for item in golden:
    correct = {item["chunk_index"], *EXTRA_CORRECT.get(item["id"], [])}
    question_vector = embed(["search_query: " + item["question"]])[0]
    closeness = chunk_vectors @ question_vector  # one score per chunk
    ranking = [int(i) for i in np.argsort(-closeness)]  # best match first
    rank = min(ranking.index(c) for c in correct) + 1  # 1 = the correct chunk came first
    rows.append({
        "id": item["id"],
        "correct_chunks": sorted(correct),
        "rank": rank,
        "top_chunks": ranking[:K],
    })
    print(f"Q{item['id']}: correct chunk(s) {sorted(correct)} ranked #{rank} | top {K}: {ranking[:K]}")

hit_at_1 = sum(r["rank"] == 1 for r in rows) / len(rows)
hit_at_k = sum(r["rank"] <= K for r in rows) / len(rows)
mrr = sum(1 / r["rank"] for r in rows) / len(rows)

print(f"\nCorrect chunk ranked #1: {hit_at_1:.0%}")
print(f"Correct chunk in top {K}: {hit_at_k:.0%}")
print(f"MRR (average of 1/rank, 1.0 is perfect): {mrr:.2f}")

json.dump(rows, open("data/retrieval_results.json", "w"), indent=2)
# ---- Part 2: Noise Sensitivity (how much do wrong paragraphs change the answers?) ----
import os

if os.path.exists("data/answers_noisy.json"):
    clean = json.load(open("data/answers.json"))
    noisy = json.load(open("data/answers_noisy.json"))
    expected = {item["id"]: item["expected_answer"] for item in golden}

    def closeness_to_expected(answer, question_id):
        a, e = embed([answer, expected[question_id]])
        return float(a @ e)  # 1.0 means the same meaning

    drops = {}
    print("\nNoise sensitivity (a bigger drop means the noise hurt more)")
    for key in noisy:
        if key not in clean:
            continue
        model = noisy[key]["model"]
        qid = noisy[key]["question_id"]
        before = closeness_to_expected(clean[key]["answer"], qid)
        after = closeness_to_expected(noisy[key]["answer"], qid)
        drops.setdefault(model, []).append(before - after)
        print(f"  {key}: clean {before:.2f} -> noisy {after:.2f} (drop {before - after:+.2f})")

    for model, values in drops.items():
        print(f"{model}: average drop {sum(values) / len(values):+.3f} over {len(values)} questions")