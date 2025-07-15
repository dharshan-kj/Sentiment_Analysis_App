# src/utils.py

# Mapping from class ID to sentiment label
id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# You can also add the reverse mapping if needed
label2id = {v: k for k, v in id2label.items()}
