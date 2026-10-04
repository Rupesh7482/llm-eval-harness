import pandas as pd
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from judge import OllamaJudge

df = pd.read_csv("data/latency_raw.csv")
judge = OllamaJudge()

for _, row in df[df["question_id"] == 1].iterrows():
    case = LLMTestCase(input=row["prompt"], actual_output=row["answer"])
    metric = AnswerRelevancyMetric(model=judge, threshold=0.5, include_reason=True)
    metric.measure(case)
    print(row["model"], "| score:", metric.score)
    print("  reason:", metric.reason)