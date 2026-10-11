from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from judge import OllamaJudge

judge = OllamaJudge()
question = "What is a RAG system? Answer in two sentences."

answers = {
    "correct": "Retrieval-Augmented Generation (RAG) retrieves relevant documents and gives them to a language model so its answer is grounded in them.",
    "on-topic but wrong": "RAG stands for Red, Amber, Green, a color-coding system for project status.",
    "off-topic": "The capital of France is Paris, and it is famous for the Eiffel Tower.",
}

for label, answer in answers.items():
    case = LLMTestCase(input=question, actual_output=answer)
    metric = AnswerRelevancyMetric(model=judge, threshold=0.5, include_reason=True)
    metric.measure(case)
    print(label, "| score:", metric.score)