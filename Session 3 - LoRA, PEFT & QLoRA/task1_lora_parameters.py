from pathlib import Path
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

def evaluate_lora_parameter_efficiency(model_id: str = "distilbert-base-uncased"):
    print("=" * 70)
    print("TASK 1: HUGGING FACE TRANSFORMERS + PEFT LORA PARAMETER ANALYSIS")
    print("=" * 70)
    print(f"Loading pre-trained base model: '{model_id}'...")

    base_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    original_total_params = sum(p.numel() for p in base_model.parameters())
    original_trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    print(f"Base Model Total Parameters:      {original_total_params:,}")
    print(f"Base Model Trainable Parameters:  {original_trainable_params:,} (100.00% trainable)\n")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        bias="none"
    )

    peft_model = get_peft_model(base_model, lora_config)
    lora_total_params = sum(p.numel() for p in peft_model.parameters())
    lora_trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    trainable_pct = (lora_trainable_params / lora_total_params) * 100.0

    print("--- Applying Parameter-Efficient Fine-Tuning (LoRA) ---")
    print(f"LoRA Rank (r): {lora_config.r} | Alpha: {lora_config.lora_alpha} | Dropout: {lora_config.lora_dropout}")
    print(f"Target Attention Modules: {lora_config.target_modules}")
    print(f"Model Total Parameters (Base + Adapter): {lora_total_params:,}")
    print(f"Trainable Parameters (Adapter Weights):  {lora_trainable_params:,}")
    print(f"Trainable Parameters Ratio:              {trainable_pct:.4f}%")
    print(f"Parameter Reduction:                     {100.0 - trainable_pct:.2f}% parameters frozen")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_lora_parameter_efficiency()
