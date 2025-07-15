import streamlit as st
from src.inference import predict

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("Sentiment Analysis Web App")

st.write(
    "Enter a text below and click **Analyze** to predict whether the sentiment is Positive, Negative, or Neutral."
)

user_input = st.text_area("Your text here:")

if st.button("Analyze"):
    if user_input.strip():
        label = predict(user_input)
        if label == "positive":
            st.success(f"Predicted Sentiment: {label.capitalize()} 😊")
        elif label == "negative":
            st.error(f"Predicted Sentiment: {label.capitalize()} 😞")
        else:
            st.info(f"Predicted Sentiment: {label.capitalize()} 😐")
    else:
        st.warning("Please enter some text to analyze.")
    