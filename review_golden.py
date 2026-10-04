import json

items = json.load(open("data/golden_draft.json"))
keep = []

for item in items:
    print("=" * 70)
    print("QUESTION:", item["question"])
    print("EXPECTED ANSWER:", item["expected_answer"])
    print("\nCONTEXT:\n" + item["context"])
    choice = input("\nKeep this one? (y/n): ").strip().lower()
    if choice == "y":
        keep.append(item)

with open("data/golden.json", "w") as f:
    json.dump(keep, f, indent=2)
print(len(keep), "items saved to data/golden.json")