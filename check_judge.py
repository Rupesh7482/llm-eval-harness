import json

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

for file, name in [
    ("data/scores_faithfulness.json", "llama3.2 (3B) judge"),
    ("data/scores_faithfulness_8b.json", "llama3.1 (8B) judge"),
]:
    scores = json.load(open(file))
    agree = 0
    print("\n" + name)
    for key, human in labels.items():
        judge = scores[key]["faithfulness"] >= 0.5  # 0.5 or more counts as "faithful"
        if judge == human:
            agree += 1
        else:
            print("  disagrees on", key, "| human:", human,
                  "| judge score:", round(scores[key]["faithfulness"], 2))
    print(f"  agreement: {agree} of {len(labels)}")