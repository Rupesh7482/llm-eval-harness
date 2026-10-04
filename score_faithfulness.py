import json
import os

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from judge import OllamaJudge

JUDGE_NAME = "llama3.1:8b"  # bigger judge; the first run used llama3.2 (3B)
OUT = "data/scores_faithfulness_8b.json"  # separate file, old 3B scores stay untouched

golden = {item["id"]: item for item in json.load(open("data/golden.json"))}
answers = json.load(open("data/answers.json"))
judge = OllamaJudge(JUDGE_NAME)

scores = json.load(open(OUT)) if os.path.exists(OUT) else {}

for key, ans in answers.items():
    if key in scores:
        continue  # already marked, skip
    item = golden[ans["question_id"]]
    case = LLMTestCase(
        input=item["question"],
        actual_output=ans["answer"],
        retrieval_context=[item["context"]],  # the paragraph the model was given
    )
    metric = FaithfulnessMetric(model=judge, threshold=0.5, include_reason=True)
    metric.measure(case)
    scores[key] = {
        "model": ans["model"],
        "question_id": ans["question_id"],
        "faithfulness": metric.score,
        "reason": metric.reason,
    }
    with open(OUT, "w") as f:
        json.dump(scores, f, indent=2)  # saved after every item
    print(ans["model"], "| Q", ans["question_id"], "| score:", metric.score)

print("\nJudge:", JUDGE_NAME)
print("Average per model:")
for model in sorted({s["model"] for s in scores.values()}):
    values = [s["faithfulness"] for s in scores.values() if s["model"] == model]
    print(" ", model, round(sum(values) / len(values), 2))