import json
import os
import sys
import time

from google.genai import errors

from evaluate import ask_gemini, ask_ollama

# run as:  python run_exam.py          (clean exam)
#          python run_exam.py noisy    (exam with wrong paragraphs mixed in)
MODE = sys.argv[1] if len(sys.argv) > 1 else "clean"

golden = json.load(open("data/golden.json"))
chunks = json.load(open("data/chunks.json"))
retrieval = {r["id"]: r for r in json.load(open("data/retrieval_results.json"))}

CACHE = "data/answers.json" if MODE == "clean" else "data/answers_noisy.json"
answers = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.
Keep the answer to 1-3 sentences.

CONTEXT:
{context}

QUESTION: {question}"""

MAX_TRIES = 6
RETRY_CODES = (429, 500, 502, 503, 504)  # overloaded / rate limited / temporary


def get_context(item):
    if MODE == "clean":
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
            # NEW: a daily-quota 429 will not clear by waiting, so stop retrying at once
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


MODELS = [("llama3.2 (Ollama)", ask_ollama), ("gemini (cloud)", ask_gemini)]

for name, fn in MODELS:
    skipped = []
    for item in golden:
        key = f"{name}|{item['id']}"
        if key in answers:
            continue  # already answered, never spend a call twice

        prompt = PROMPT.format(context=get_context(item), question=item["question"])
        result, retries = ask_with_retry(name, fn, prompt, item["id"])

        if result is None:
            skipped.append(item["id"])
            continue  # skip this question, keep going with the rest

        answers[key] = {"model": name, "question_id": item["id"], "retries": retries, **result}
        with open(CACHE, "w") as f:
            json.dump(answers, f, indent=2)  # saved after every answer
        print(name, "answered question", item["id"], "in", result["total_ms"], "ms",
              f"(retries: {retries})")

        if "gemini" in name:
            time.sleep(2)  # small gap so we don't trigger the rate limit

    if skipped:
        print(f"{name}: skipped questions {skipped}. Run this script again to fill them in.")

print(MODE, "mode:", len(answers), "answers saved in total")