from src.data_loader import load_data
from src.model import get_model
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from src.train import TextDataset

def evaluate():
    # Load test data
    test_texts, test_labels = load_data("data/test.csv")
    
    # Load fine-tuned model and tokenizer from saved directory
    model, tokenizer = get_model(model_path="model_save/")
    model.eval()
    
    # Prepare dataset and dataloader
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Print classification report with label names matching your 3-class setup
    print(classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"]))

if __name__ == "__main__":
    evaluate()
