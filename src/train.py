import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
from src.data_loader import load_data
from src.model import get_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bar

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load data and split into train and validation
    texts, labels = load_data("data/train.csv")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

    model, tokenizer = get_model()
    model.to(device)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)

    # Learning rate scheduler (linear decay)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        loop = tqdm(train_loader, leave=True)
        total_loss = 0
        for batch in loop:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            loop.set_description(f"Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluate on validation set after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")
        model.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("model_save/")
    tokenizer.save_pretrained("model_save/")

if __name__ == "__main__":
    train()
