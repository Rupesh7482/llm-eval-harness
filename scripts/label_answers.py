import json
import os

golden = {item["id"]: item for item in json.load(open("data/golden.json"))}
answers = json.load(open("data/answers.json"))

OUT = "data/human_labels.json"
labels = json.load(open(OUT)) if os.path.exists(OUT) else {}

for key, ans in answers.items():
    if key in labels:
        continue  # already labelled
    item = golden[ans["question_id"]]
    print("=" * 70)
    print("CONTEXT:\n" + item["context"])
    print("\nQUESTION:", item["question"])
    print("\nANSWER:", ans["answer"])
    choice = input("\nIs EVERY claim in the answer stated in the context? (y/n, q=quit): ")
    choice = choice.strip().lower()
    if choice == "q":
        break
    labels[key] = (choice == "y")
    with open(OUT, "w") as f:
        json.dump(labels, f, indent=2)  # saved after every answer

print(sum(labels.values()), "of", len(labels), "answers marked faithful")
