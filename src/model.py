from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer
