import pandas as pd
from src.utils import label2id

def load_data(filepath):
    df = pd.read_csv(r'C:\Users\Dharshan K J\OneDrive\Desktop\increff_ass\sentiment_app\data\train.csv')
    texts = df['text'].tolist()
    labels = [label2id[label] for label in df['label'].tolist()]
    return texts, labels
