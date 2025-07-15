# src/inference.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.utils import id2label

model_name = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict(text):
    """
    Predict sentiment label for a given text.
    Returns one of 'negative', 'neutral', 'positive'.
    """

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred_class_id = torch.argmax(logits, dim=1).item()
    return id2label.get(pred_class_id, "unknown")
