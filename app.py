import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LLM Evaluation Harness", layout="wide")
st.title("LLM Evaluation Harness")
st.caption("Llama 3.2 (local, via Ollama) vs Gemini (cloud) answering questions from one document.")
st.warning(
    "Small experiment: 9 questions from one document, one run. "
    "The differences are small and should not be treated as proof."
)

df = pd.read_csv("data/results_real.csv")
df["question"] = "Q" + df["question_id"].astype(str)
order = {"question": ["Q" + str(i) for i in sorted(df["question_id"].unique())]}

# one row per model, averaged over the questions
summary = df.groupby("model").agg(
    answer_similarity=("answer_similarity", "mean"),
    faithful=("faithful", "sum"),
    questions=("faithful", "count"),
    noise_drop=("noise_drop", "mean"),
    median_latency_ms=("latency_ms", "median"),
).reset_index()

st.subheader("Summary")
columns = st.columns(len(summary))
for column, (_, row) in zip(columns, summary.iterrows()):
    with column:
        st.markdown(f"**{row['model']}**")
        st.metric("Answer similarity to expected answer (0 to 1)", f"{row['answer_similarity']:.2f}")
        st.metric("Answers with nothing made up", f"{int(row['faithful'])} of {int(row['questions'])}")
        st.metric("Noise drop (positive = noise hurt)", f"{row['noise_drop']:+.3f}")
        st.metric("Median response time", f"{row['median_latency_ms'] / 1000:.1f} s")

left, right = st.columns(2)
with left:
    st.subheader("Answer similarity per question")
    st.plotly_chart(px.bar(df, x="question", y="answer_similarity", color="model",
                           barmode="group", category_orders=order))
with right:
    st.subheader("Response time per question (ms)")
    st.plotly_chart(px.bar(df, x="question", y="latency_ms", color="model",
                           barmode="group", category_orders=order))

st.subheader("Search step: did the right paragraph rank first?")
retrieval = df.drop_duplicates("question_id")[["question", "retrieval_rank"]]
first = (retrieval["retrieval_rank"] == 1).mean()
top3 = (retrieval["retrieval_rank"] <= 3).mean()
st.write(f"The correct paragraph ranked first for {first:.0%} of questions and in the top 3 for {top3:.0%}.")
st.plotly_chart(px.bar(retrieval, x="question", y="retrieval_rank", category_orders=order))
st.caption("Rank 1 is best. Both models received the correct paragraph, so this measures the search step only.")

with st.expander("Limitations"):
    st.markdown(
        "- The answer key was drafted by Llama, then corrected by hand.\n"
        "- 'Nothing made up' comes from my own reading of each answer.\n"
        "- Four AI judges were tested for faithfulness and rejected, because they disagreed with my labels.\n"
        "- Similarity measures closeness of wording and meaning, not truth.\n"
        "- Gemini's speed depends on how busy Google's servers are.\n"
        "- The noise test was gentle: two wrong paragraphs placed before the right one."
    )

st.subheader("All results")
st.dataframe(df.drop(columns=["question"]))