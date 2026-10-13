# LLM Evaluation Harness: Llama 3.2 (local) vs Gemini (cloud)

A small experiment that compares a local model (Llama 3.2 via Ollama) and a cloud model (Gemini)
on answering questions from one document. It measures answer quality, made-up content, speed,
and how well the search step finds the right paragraph. Results are shown on a Streamlit dashboard.

**Live dashboard:** https://llm-eval-harness.streamlit.app

> This is a small experiment: 9 questions, one document, one run. The differences are small
> and should not be treated as proof.

## Results

| | Gemini | Llama 3.2 |
|---|---|---|
| Answer similarity to the expected answer (0 to 1) | 0.90 | 0.88 |
| Answers with nothing made up (my own labels) | 9 of 9 | 7 of 9 |
| Noise drop (positive = wrong paragraphs hurt) | -0.011 | -0.015 |
| Median response time | about 3.1 s | about 1.3 s |

**Search step** (same for both models): the correct paragraph ranked first for 6 of 9 questions (67%),
in the top 3 for 8 of 9 (89%), MRR 0.79. One question (Q2, about natural acceptance) ranked 8th.

## How it works

1. **Chunks:** the PDF is cut into 29 paragraphs of 800 characters (100 characters overlap).
2. **Answer key:** Llama drafted a question and answer for 10 paragraphs. I checked each one against
   the source text, fixed two wrong ones, and dropped one that made no sense alone. 9 questions remain.
3. **Exam:** both models answer every question using only the correct paragraph (clean exam).
4. **Noisy exam:** the same, but two wrong paragraphs are added before the right one. They are the
   two highest-ranked wrong results from the search.
5. **Search test:** each question is compared with all 29 paragraphs using the `nomic-embed-text`
   model, and I check where the correct paragraph lands.
6. **Scoring:** answer similarity is the closeness in meaning to my expected answer (same embedding model).
   "Nothing made up" comes from my own reading of each answer.

## What did not work: AI judges for faithfulness

I tested four ways of letting a local model judge whether an answer made things up, and compared each
with my own labels (18 answers, 2 of them made-up):

| Judge | Agreed with me | Made-up answers caught | Good answers wrongly flagged |
|---|---|---|---|
| DeepEval Faithfulness, Llama 3.2 (3B) | 8 of 18 | 0 of 2 | 8 |
| DeepEval Faithfulness, Llama 3.1 (8B) | 10 of 18 | 1 of 2 | 7 |
| Simple YES/NO prompt, 8B | 16 of 18 | 0 of 2 | 0 |
| "List unsupported facts" prompt, 8B | 2 of 18 | 2 of 2 | 16 |

None caught made-up answers without flagging good ones, and the 3B judge gave different scores
to the same input on two runs. So the dashboard uses my own labels instead.
These judge experiments are kept in `experiments/`.

## Limitations

- Only 9 questions from one document, one run per model.
- The answer key was drafted by Llama, then corrected by hand.
- "Nothing made up" is my own reading. I re-read the paragraphs for two Llama answers and
  corrected those labels.
- Answer similarity measures closeness of wording and meaning, not truth.
- The noise test was gentle: two wrong paragraphs placed before the right one.
- Gemini's speed depends on how busy Google's servers are (7 to 45 seconds in some runs).
  The free tier is also limited to about 20 requests a day per model.
- The search test uses one embedding model, and the questions were written from the paragraphs,
  which makes search look easier than with real users.

## Project structure

```
app.py            Streamlit dashboard (reads data/results_real.csv)
models.py         calls Llama (Ollama) and Gemini and measures response time
pipeline.py       the evaluation steps: chunks, exam, retrieval, results
experiments/      the AI-judge attempts that did not work
scripts/          one-time work: drafting and fixing the answer key, labelling answers
data/             results_real.csv, retrieval_results.json, human_labels.json, latency_raw.csv
requirements.txt        packages the dashboard needs
requirements-eval.txt   extra packages needed to run the evaluation
```

The source PDF, the paragraphs, the answer key and the saved answers are not in this repository
(the PDF is course material). Only numbers and rankings are published.

## Run it yourself

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-eval.txt
ollama pull llama3.2
ollama pull nomic-embed-text
echo "GOOGLE_API_KEY=your_key" > .env
streamlit run app.py        # shows the dashboard from the published results
```

To rerun the whole evaluation you need your own PDF at `data/source.pdf` and your own answer key.
Then:

```
python pipeline.py chunks
python pipeline.py exam
python pipeline.py exam noisy
python pipeline.py retrieval
python pipeline.py results
```

Answers are saved after every call, so rerunning never repeats a request.

## File by file

- **app.py**: builds the dashboard. It reads `results_real.csv`, shows one summary card per model,
  per-question charts, the search chart, the limitations, and the full table.
- **models.py**: one function per model (`ask_ollama`, `ask_gemini`). Each streams the answer,
  times the first token and the total, and returns the answer with its timings.
- **pipeline.py**: four steps run from the command line. `chunks` cuts the PDF, `exam` asks both models
  every question (with retries when Google is busy), `retrieval` ranks paragraphs for each question,
  and `results` collects every number into one table.
- **experiments/**: `judge.py` lets DeepEval use a local model as judge, `score_faithfulness.py` and
  `check_judge.py` run and test the judges, `first_metric.py` and `sanity_check.py` were first tests.
- **scripts/**: `make_golden_draft.py` drafts questions, `review_golden.py` is the yes/no review,
  `fix_golden.py` applies my corrections, `label_answers.py` records my faithful/not-faithful labels,
  `analyze_latency.py` summarises the first speed test.