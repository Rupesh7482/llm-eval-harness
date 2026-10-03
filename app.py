import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LLM Evaluation Harness", layout="wide")
st.title("LLM Evaluation Harness")
st.info("Demo data for now. Real evaluation results will replace it.")

df = pd.read_csv("data/results.csv")
metrics = ["answer_relevancy", "faithfulness", "context_precision", "noise_sensitivity"]

# average of every metric per model
summary = df.groupby("model")[metrics + ["latency_ms"]].mean().reset_index()

st.subheader("Average score per metric")
long_form = summary.melt(id_vars="model", value_vars=metrics,
                         var_name="metric", value_name="score")
st.plotly_chart(px.bar(long_form, x="metric", y="score",
                       color="model", barmode="group"))

st.subheader("Average latency (ms)")
st.plotly_chart(px.bar(summary, x="model", y="latency_ms", color="model"))

st.subheader("Raw results")
st.dataframe(df)