from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

class TweetSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def fine_tune_lora_model():
    print("=" * 70)
    print("TASK 2: FINE-TUNING TEXT CLASSIFICATION MODEL WITH LORA & PEFT")
    print("=" * 70)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    sample_tweets = [
        "Absolutely loved the swift delivery and piping hot biryani from Zomato!",
        "Incredible service, food arrived 15 minutes before the estimated time!",
        "The packaging was pristine and the dessert tasted heavenly.",
        "Super responsive customer support, resolved my missing item query instantly.",
        "Delicious authentic Gujarati thali, five star dining experience at home!",
        "Worst food delivery experience ever, soup arrived completely spilled.",
        "Order was delayed by over an hour and arrived freezing cold, terrible service.",
        "Support bot was unhelpful and kept looping without offering any refund.",
        "Wrong order delivered, highly disappointed with the callous attitude.",
        "Food was stale and undercooked, definitely will never order again."
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    dataset = TweetSentimentDataset(sample_tweets, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        bias="none"
    )

    peft_model = get_peft_model(base_model, lora_config)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)

    print("Beginning LoRA fine-tuning training loop (3 epochs)...")
    peft_model.train()
    for epoch in range(1, 4):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            batch_labels = batch["labels"]
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/3 - Training Loss: {avg_loss:.4f}")

    output_dir = Path(__file__).resolve().parent / "lora_adapted_weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nAdapted LoRA weights saved successfully to: '{output_dir}'")
    print("Saved Artifact File Inspection:")
    total_size_bytes = 0
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            size_kb = file_path.stat().st_size / 1024.0
            total_size_bytes += file_path.stat().st_size
            print(f"- {file_path.name:30} : {size_kb:8.2f} KB")

    total_size_mb = total_size_bytes / (1024.0 * 1024.0)
    print(f"Total Adapter Directory Footprint: {total_size_mb:.2f} MB ({total_size_bytes:,} bytes)")
    print("=" * 70)

if __name__ == "__main__":
    fine_tune_lora_model()
