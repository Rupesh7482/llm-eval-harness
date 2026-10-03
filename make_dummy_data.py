import numpy as np
import pandas as pd

rng = np.random.default_rng(42)   # fixed seed so the numbers are the same every run
models = ["llama3.2 (Ollama)", "gemini (cloud)"]

rows = []
for model in models:
    for question_id in range(1, 21):          # 20 questions
        local = model.startswith("llama")
        rows.append({
            "model": model,
            "question_id": question_id,
            "answer_relevancy": rng.uniform(0.5, 0.95),
            "faithfulness": rng.uniform(0.4, 0.9),
            "context_precision": rng.uniform(0.5, 0.95),
            "noise_sensitivity": rng.uniform(0.05, 0.4),
            "latency_ms": rng.uniform(2500, 6000) if local else rng.uniform(800, 2000),
        })

df = pd.DataFrame(rows).round(3)
df.to_csv("data/results.csv", index=False)
print(df.head())