import json
import os

import ollama

# Corrections to your labels, after re-reading the paragraphs.
OVERRIDES = {
    "gemini (cloud)|2": True,        # you typed "yy" by mistake, so it was saved as False
    "llama3.2 (Ollama)|4": False,    # answer adds "personal reflection", not in the paragraph
    "llama3.2 (Ollama)|5": False,    # answer adds "healthy and harmonious connections", not in the paragraph
}

labels = json.load(open("data/human_labels.json"))
labels.update(OVERRIDES)
json.dump(labels, open("data/human_labels.json", "w"), indent=2)
print(sum(labels.values()), "of", len(labels), "marked faithful by you")

# ---- Part 1: the earlier DeepEval judges ----
for file, name in [
    ("data/scores_faithfulness.json", "llama3.2 (3B) judge"),
    ("data/scores_faithfulness_8b.json", "llama3.1 (8B) judge"),
]:
    scores = json.load(open(file))
    agree = sum((scores[k]["faithfulness"] >= 0.5) == labels[k] for k in labels)
    print(f"{name}: agreement {agree} of {len(labels)}")

# ---- Part 2: the "list the unsupported facts" judge ----
VERDICTS = "data/verdicts_v2.json"
golden = {item["id"]: item for item in json.load(open("data/golden.json"))}
answers = json.load(open("data/answers.json"))
verdicts = json.load(open(VERDICTS)) if os.path.exists(VERDICTS) else {}

PROMPT = """CONTEXT:
{context}

ANSWER:
{answer}

Check the ANSWER against the CONTEXT.
List every fact, word or idea in the ANSWER that is NOT stated in the CONTEXT.
Ignore spelling mistakes, missing spaces, and facts that the ANSWER leaves out.
If everything in the ANSWER is stated in the CONTEXT, reply with the single word NONE.
Otherwise, list the unsupported items."""

for key in labels:
    if key in verdicts:
        continue  # already judged, skip
    ans = answers[key]
    prompt = PROMPT.format(
        context=golden[ans["question_id"]]["context"], answer=ans["answer"]
    )
    reply = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},  # same input gives the same answer every time
    )["message"]["content"]
    faithful = reply.strip().upper().startswith("NONE")
    verdicts[key] = {"faithful": faithful, "reply": reply}
    json.dump(verdicts, open(VERDICTS, "w"), indent=2)

bad = [k for k in labels if not labels[k]]
caught = [k for k in bad if not verdicts[k]["faithful"]]
false_alarms = [k for k in labels if labels[k] and not verdicts[k]["faithful"]]
agree = sum(verdicts[k]["faithful"] == labels[k] for k in labels)

print("\nList-the-unsupported-facts judge (llama3.1 8B)")
print(f"  agreement: {agree} of {len(labels)}")
print(f"  made-up answers caught: {len(caught)} of {len(bad)}")
print(f"  false alarms (good answers flagged): {len(false_alarms)}")

print("\nWhat the judge said on the cases that matter:")
for k in bad + false_alarms:
    print(" ", k, "->", verdicts[k]["reply"][:200].replace("\n", " "))